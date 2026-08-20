"""Pre-registered four-seed evidence ensemble on the V18 production recipe."""
from __future__ import annotations

import json

import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.optim import AdamW
from tqdm import tqdm

from analyze_task1_atomic_refine_v26 import _refine_one
from analyze_task1_lexical_v11 import _lexical_experts, _prediction, _transformer_probability
from baseline import _post_phrase_f1
from configs.config import config
from datasets.dataset import SuicideRiskDataset
from inference.task1_evidence_v4 import apply_evidence_policy, decode_model_evidence
from models.multitask_model import SuicideRiskMultiTaskModel, get_optimizer_parameters
from preprocess.preprocess import load_train_data
from trainer.task1_atomic_v25 import _atomic_candidates
from trainer.task1_seed_ensemble_v14 import _collect_seed, _criterion
from trainer.train import _loader, _move
from utils.seed import seed_everything
from utils.task1_metric import task1_score


OUTPUT = config.OUTPUT_DIR / "task1_seed_evidence_v28"
RESULTS = OUTPUT / "results.json"
CALIBRATION = OUTPUT / "calibration.json"
TRAINING_VERSION = "task1-v18-four-seed-evidence-v28"
EXTRA_SEEDS = (271828, 161803)


def _paths(seed):
    return OUTPUT / f"seed_{seed}_model.pt", OUTPUT / f"seed_{seed}_valid.pt"


def _train_seed(seed, dataset, train_idx, valid_idx, labels, device):
    checkpoint, prediction_file = _paths(seed)
    if checkpoint.exists() and prediction_file.exists():
        saved = torch.load(prediction_file, map_location="cpu", weights_only=False)
        if np.array_equal(np.asarray(saved["valid_idx"]), valid_idx):
            print(f"V28 resumed seed {seed}", flush=True)
            return saved["rows"], saved["history"]
    seed_everything(seed)
    model = SuicideRiskMultiTaskModel().to(device)
    model.backbone.encoder.gradient_checkpointing_disable()
    optimizer = AdamW(get_optimizer_parameters(model), weight_decay=config.WEIGHT_DECAY)
    criterion = _criterion(dataset, train_idx, labels, device)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    loader = _loader(dataset, train_idx, True); history = []
    print(f"V28 training seed {seed}: posts={len(train_idx)}, epochs=3", flush=True)
    for epoch in range(1, 4):
        model.train(); optimizer.zero_grad(set_to_none=True); losses = []
        for step, batch in enumerate(tqdm(loader, desc=f"V28 seed {seed} epoch {epoch}/3"), 1):
            batch = _move(batch, device)
            with torch.autocast(device_type="cuda", enabled=True):
                loss = criterion(model(batch["input_ids"], batch["attention_mask"]), batch)["loss"]
                loss = loss / config.GRADIENT_ACCUMULATION
            scaler.scale(loss).backward()
            losses.append(float(loss.detach()) * config.GRADIENT_ACCUMULATION)
            if step % config.GRADIENT_ACCUMULATION == 0 or step == len(loader):
                scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses))})
        print(f"V28 seed={seed} epoch={epoch} loss={history[-1]['train_loss']:.4f}", flush=True)
    torch.save(model.state_dict(), checkpoint)
    rows = _collect_seed(model, _loader(dataset, valid_idx, False), device)
    torch.save({"training_version": TRAINING_VERSION, "seed": seed, "valid_idx": valid_idx,
                "history": history, "rows": rows}, prediction_file)
    del model, optimizer, criterion, loader; torch.cuda.empty_cache()
    return rows, history


def _decode(records, risks, starts, ends, parameters):
    evidence = []
    for record, risk, start, end in zip(records, risks, starts, ends):
        policy = parameters[config.ID2RISK[int(risk)]]
        spans = decode_model_evidence(
            record["text"], record["offsets"], start, end,
            threshold=policy["threshold"], max_tokens=policy["max_tokens"],
            end_policy=policy["end_policy"], limit=5,
        )
        evidence.append(apply_evidence_policy(
            record["text"], int(risk), spans,
            policy=policy["cue_policy"], topk=policy["topk"],
        ))
    return evidence


def _scores(evidence, gold):
    return np.asarray([
        _post_phrase_f1(prediction, target)
        for prediction, target in zip(evidence, gold)
    ], dtype=np.float32)


def train_task1_seed_evidence_v28():
    if not torch.cuda.is_available():
        raise RuntimeError("V28 requires CUDA")
    OUTPUT.mkdir(parents=True, exist_ok=True); device = torch.device("cuda")
    dataset = SuicideRiskDataset(config.CACHE_DIR / "train_cache.pt")
    frame = load_train_data().reset_index(drop=True)
    labels = frame.risk_label.to_numpy(); groups = frame.anon_user_id.astype(str).to_numpy()
    train_idx, valid_idx = next(StratifiedGroupKFold(
        n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED
    ).split(np.zeros(len(frame)), labels, groups))
    raw = torch.load(config.OUTPUT_DIR / "task1_evidence_v4" / "fold0_raw.pt",
                     map_location="cpu", weights_only=False)
    if not np.array_equal(np.asarray(raw["valid_idx"]), valid_idx):
        raise ValueError("V28 strict rows differ from V18")
    records = raw["records"]
    seed2 = torch.load(
        config.OUTPUT_DIR / "task1_seed_ensemble_v14" / "seed2_valid.pt",
        map_location="cpu", weights_only=False,
    )["rows"]
    extras = []; histories = {}
    for seed in EXTRA_SEEDS:
        rows, history = _train_seed(seed, dataset, train_idx, valid_idx, labels, device)
        extras.append(rows); histories[str(seed)] = history

    # Reconstruct the exact current V18 risk labels.  The new seeds are not
    # allowed to alter risk, isolating evidence variance reduction.
    transformer = _transformer_probability(dataset, valid_idx, records)
    lexical = _lexical_experts(frame, train_idx, valid_idx)["svc-c0.25-balanced"]
    risk_parameters = {"expert": "svc-c0.25-balanced", "temperature": 1.0,
                       "lexical_weight": 0.60, "attempt_bias": 0.20}
    risks = _prediction(records, transformer, lexical, risk_parameters)
    v18 = json.loads((config.OUTPUT_DIR / "task1_candidate_v18" / "results.json").read_text())
    evidence_parameters = v18["evidence_parameters_by_predicted_risk"]
    main_start = [row["start"] for row in records]; main_end = [row["end"] for row in records]
    seed2_start = [row["start"] for row in seed2]; seed2_end = [row["end"] for row in seed2]
    baseline_start = [0.8 * a + 0.2 * b for a, b in zip(main_start, seed2_start)]
    baseline_end = [0.8 * a + 0.2 * b for a, b in zip(main_end, seed2_end)]
    baseline_evidence = _decode(records, risks, baseline_start, baseline_end, evidence_parameters)

    auxiliary_sets = [seed2] + extras
    candidate_start = [];
    candidate_end = []
    for index in range(len(records)):
        mean_start = sum(rows[index]["start"] for rows in auxiliary_sets) / len(auxiliary_sets)
        mean_end = sum(rows[index]["end"] for rows in auxiliary_sets) / len(auxiliary_sets)
        candidate_start.append(0.8 * main_start[index] + 0.2 * mean_start)
        candidate_end.append(0.8 * main_end[index] + 0.2 * mean_end)
    candidate_evidence = _decode(records, risks, candidate_start, candidate_end, evidence_parameters)

    # V26's policy was selected without these new seeds and is fixed before
    # this outer evaluation, so it can be safely composed with the ensemble.
    boundary_file = config.OUTPUT_DIR / "task1_atomic_refine_v26" / "results.json"
    atomic_file = config.OUTPUT_DIR / "task1_atomic_refine_v26" / "atomic_outputs.pt"
    if boundary_file.exists() and atomic_file.exists():
        boundary = json.loads(boundary_file.read_text())["selected"]
        outputs = torch.load(atomic_file, map_location="cpu", weights_only=False)["outer_outputs"]
        refined = []
        for global_index, evidence, risk in zip(valid_idx, candidate_evidence, risks):
            text = str(frame.iloc[int(global_index)].text)
            atomic = _atomic_candidates(
                text, outputs.get(int(global_index), []), boundary["token_threshold"],
                boundary["sentence_threshold"], boundary["max_tokens"],
            )
            refined.append([] if int(risk) == 0 else _refine_one(text, evidence, atomic, boundary))
        candidate_evidence = refined

    gold = [record["gold"] for record in records]
    baseline_phrase = _scores(baseline_evidence, gold)
    candidate_phrase = _scores(candidate_evidence, gold)
    risk_f1 = float(f1_score(labels[valid_idx], risks, average="weighted", zero_division=0))
    baseline_task = task1_score(risk_f1, baseline_phrase.mean())
    candidate_task = task1_score(risk_f1, candidate_phrase.mean())
    unique = np.unique(groups[valid_idx]); rng = np.random.default_rng(config.SEED + 2828)
    deltas = []
    for _ in range(5000):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        positions = np.concatenate([np.flatnonzero(groups[valid_idx] == user) for user in sampled])
        deltas.append(float(candidate_phrase[positions].mean() - baseline_phrase[positions].mean()))
    deltas = np.asarray(deltas)
    bootstrap = {"mean_phrase_delta": float(deltas.mean()),
                 "p05_phrase_delta": float(np.quantile(deltas, .05)),
                 "p95_phrase_delta": float(np.quantile(deltas, .95)),
                 "positive_fraction": float((deltas > 0).mean())}
    adopted = bool(candidate_task >= baseline_task + .003 and bootstrap["positive_fraction"] >= .80)
    payload = {
        "training_version": TRAINING_VERSION,
        "evaluation_scope": "fixed four-seed recipe; one strict outer-user evaluation",
        "pre_registered_recipe": {"main_weight": .8, "auxiliary_total_weight": .2,
                                  "auxiliary_seeds": [31415, *EXTRA_SEEDS],
                                  "risk_unchanged": True, "v26_boundary": True},
        "histories": histories,
        "baseline_v18": {"risk_f1": risk_f1, "phrase_f1": float(baseline_phrase.mean()),
                         "task1": baseline_task},
        "candidate": {"risk_f1": risk_f1, "phrase_f1": float(candidate_phrase.mean()),
                      "task1": candidate_task,
                      "improved_posts": int((candidate_phrase > baseline_phrase).sum()),
                      "worsened_posts": int((candidate_phrase < baseline_phrase).sum())},
        "user_cluster_bootstrap": bootstrap, "adopted": adopted,
    }
    RESULTS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    CALIBRATION.write_text(json.dumps({"training_version": TRAINING_VERSION,
        "adopted": adopted, "strict_task1": candidate_task,
        "bootstrap_positive_fraction": bootstrap["positive_fraction"],
        "main_weight": .8, "auxiliary_total_weight": .2,
        "auxiliary_seeds": [31415, *EXTRA_SEEDS]}, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True); return payload


if __name__ == "__main__":
    train_task1_seed_evidence_v28()
