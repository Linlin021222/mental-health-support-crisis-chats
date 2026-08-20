"""A fixed two-seed DeBERTa ensemble evaluated on the strict user holdout."""
from __future__ import annotations

import json

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.svm import LinearSVC
from torch.optim import AdamW
from tqdm import tqdm

from analyze_task1_v2_ensemble import collect
from baseline import _post_phrase_f1, _vectorizer
from configs.config import config
from datasets.dataset import SuicideRiskDataset
from inference.task1_evidence_v4 import (
    apply_evidence_policy, correct_risk_only, decode_model_evidence,
    load_evidence_calibration,
)
from models.losses import MultiTaskLoss
from models.multitask_model import SuicideRiskMultiTaskModel, get_optimizer_parameters
from models.multitask_model_v2 import SuicideRiskMultiTaskModelV2
from preprocess.preprocess import load_train_data
from trainer.train import _loader, _move
from utils.seed import seed_everything
from utils.task1_metric import task1_score


OUTPUT = config.OUTPUT_DIR / "task1_seed_ensemble_v14"
CHECKPOINT = OUTPUT / "seed2_model.pt"
PREDICTIONS = OUTPUT / "seed2_valid.pt"
RESULTS = OUTPUT / "results.json"
CALIBRATION = OUTPUT / "calibration.json"
TRAINING_VERSION = "task1-two-seed-ensemble-v14"
SEED2 = 31415


def _criterion(dataset, train_idx, labels, device):
    counts = np.bincount(labels[train_idx], minlength=config.NUM_RISK_CLASSES)
    class_weights = np.sqrt(len(train_idx) / np.maximum(counts, 1))
    class_weights = torch.tensor(
        class_weights / class_weights.mean(), dtype=torch.float, device=device
    )
    factor_labels = torch.stack([
        dataset.data[int(i)]["factor_vector"] for i in train_idx
    ]).float()
    positive = factor_labels.sum(0)
    factor_weight = torch.sqrt(
        (len(train_idx) - positive) / positive.clamp_min(1.0)
    ).clamp(1.0, 10.0).to(device)
    return MultiTaskLoss(
        risk_class_weights=class_weights, factor_pos_weight=factor_weight
    ).to(device)


@torch.no_grad()
def _collect_seed(model, loader, device):
    model.eval(); rows = []
    for batch in tqdm(loader, desc="v14 seed2 strict inference", leave=False):
        metadata = batch
        output = model(
            batch["input_ids"].to(device), batch["attention_mask"].to(device)
        )
        probability = torch.softmax(output["risk_logits"], -1).cpu().numpy()
        for i in range(len(metadata["row_id"])):
            rows.append({
                "row_id": str(metadata["row_id"][i]),
                "probability": probability[i],
                "start": output["start_logits"][i].cpu(),
                "end": output["end_logits"][i].cpu(),
            })
    return rows


def _train_seed2(dataset, train_idx, valid_idx, labels, device):
    if CHECKPOINT.exists() and PREDICTIONS.exists():
        print("v14: resumed second seed checkpoint and strict logits", flush=True)
        return torch.load(PREDICTIONS, map_location="cpu", weights_only=False)
    seed_everything(SEED2)
    model = SuicideRiskMultiTaskModel().to(device)
    optimizer = AdamW(get_optimizer_parameters(model), weight_decay=config.WEIGHT_DECAY)
    criterion = _criterion(dataset, train_idx, labels, device)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    train_loader = _loader(dataset, train_idx, True)
    print(
        f"v14: training independent seed {SEED2} on {len(train_idx)} posts for 3 epochs",
        flush=True,
    )
    history = []
    for epoch in range(1, 4):
        model.train(); optimizer.zero_grad(set_to_none=True); losses = []
        progress = tqdm(train_loader, desc=f"v14 seed2 epoch {epoch}/3")
        for step, batch in enumerate(progress, 1):
            batch = _move(batch, device)
            with torch.autocast(device_type="cuda", enabled=True):
                loss = criterion(
                    model(batch["input_ids"], batch["attention_mask"]), batch
                )["loss"] / config.GRADIENT_ACCUMULATION
            scaler.scale(loss).backward()
            losses.append(float(loss.detach()) * config.GRADIENT_ACCUMULATION)
            if step % config.GRADIENT_ACCUMULATION == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
            if step % 100 == 0:
                progress.set_postfix(loss=f"{np.mean(losses[-100:]):.3f}")
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses))})
        print(f"v14 seed2 epoch={epoch} train_loss={history[-1]['train_loss']:.4f}", flush=True)
    torch.save(model.state_dict(), CHECKPOINT)
    rows = _collect_seed(model, _loader(dataset, valid_idx, False), device)
    payload = {"rows": rows, "history": history, "valid_idx": valid_idx}
    torch.save(payload, PREDICTIONS)
    del model, optimizer
    torch.cuda.empty_cache()
    return payload


def _softmax(values, temperature=0.5):
    values = np.asarray(values, dtype=np.float64) / float(temperature)
    values -= values.max(1, keepdims=True); values = np.exp(values)
    return values / values.sum(1, keepdims=True)


def _stable_components(dataset, train_idx, valid_idx, old_records, device):
    print("v14: collecting existing ordinal expert...", flush=True)
    model = SuicideRiskMultiTaskModelV2().to(device)
    model.load_state_dict(torch.load(
        config.OUTPUT_DIR / "task1_v2_strict_model.pt", map_location=device
    ))
    v2_rows = collect(model, dataset, valid_idx, device, v2=True)
    del model; torch.cuda.empty_cache()
    v2 = np.vstack([
        0.75 * row["standard"] + 0.25 * row["ordinal"] for row in v2_rows
    ])
    frame = load_train_data().reset_index(drop=True)
    vectorizer = _vectorizer()
    train_matrix = vectorizer.fit_transform(frame.text.iloc[train_idx])
    valid_matrix = vectorizer.transform(frame.text.iloc[valid_idx])
    lexical_model = LinearSVC(C=1.0, class_weight="balanced").fit(
        train_matrix, np.asarray(frame.risk_label)[train_idx]
    )
    lexical = _softmax(lexical_model.decision_function(valid_matrix), 0.5)
    old = np.vstack([row["old_probability"] for row in old_records])
    return old, v2, lexical


def _risk_predictions(records, probability):
    return np.asarray([
        correct_risk_only(record["text"], int(np.argmax(row_probability)))
        for record, row_probability in zip(records, probability)
    ], dtype=np.int64)


def _evidence_scores(records, risks, starts, ends, calibration):
    scores = np.empty(len(records), dtype=np.float32)
    for i, record in enumerate(records):
        phrases = decode_model_evidence(
            record["text"], record["offsets"], starts[i], ends[i],
            threshold=float(calibration["threshold"]),
            max_tokens=int(calibration["max_tokens"]),
            end_policy=calibration["end_policy"], limit=5,
        )
        evidence = apply_evidence_policy(
            record["text"], int(risks[i]), phrases,
            policy=calibration["cue_policy"], topk=int(calibration["topk"]),
        )
        scores[i] = _post_phrase_f1(evidence, record["gold"])
    return scores


def train_task1_seed_ensemble_v14():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("V14 requires CUDA")
    dataset = SuicideRiskDataset(config.CACHE_DIR / "train_cache.pt")
    labels = np.asarray([int(row["risk_label"]) for row in dataset.data])
    groups = np.asarray([str(row["anon_user_id"]) for row in dataset.data])
    train_idx, valid_idx = next(StratifiedGroupKFold(
        n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED
    ).split(np.zeros(len(dataset)), labels, groups))
    raw = torch.load(
        config.OUTPUT_DIR / "task1_evidence_v4" / "fold0_raw.pt",
        map_location="cpu", weights_only=False,
    )
    if not np.array_equal(raw["valid_idx"], valid_idx):
        raise ValueError("V14 and V4 strict folds differ")
    device = torch.device("cuda")
    seed2_payload = _train_seed2(dataset, train_idx, valid_idx, labels, device)
    seed2 = seed2_payload["rows"]
    if [row["row_id"] for row in seed2] != [str(row["row_id"]) for row in raw["records"]]:
        raise ValueError("Seed model strict rows are misaligned")
    old, v2, lexical = _stable_components(
        dataset, train_idx, valid_idx, raw["records"], device
    )
    new = np.vstack([row["probability"] for row in seed2])
    old_stable = 0.7 * (0.8 * old + 0.2 * v2) + 0.3 * lexical
    seed_mean = 0.5 * old + 0.5 * new
    ensemble_probability = 0.7 * (0.8 * seed_mean + 0.2 * v2) + 0.3 * lexical
    baseline_risk = _risk_predictions(raw["records"], old_stable)
    ensemble_risk = _risk_predictions(raw["records"], ensemble_probability)
    truth = labels[valid_idx]
    baseline_risk_f1 = float(f1_score(
        truth, baseline_risk, average="weighted", zero_division=0
    ))
    ensemble_risk_f1 = float(f1_score(
        truth, ensemble_risk, average="weighted", zero_division=0
    ))
    calibration = load_evidence_calibration()
    old_start = [row["start"] for row in raw["records"]]
    old_end = [row["end"] for row in raw["records"]]
    averaged_start = [
        0.5 * row["start"] + 0.5 * other["start"]
        for row, other in zip(raw["records"], seed2)
    ]
    averaged_end = [
        0.5 * row["end"] + 0.5 * other["end"]
        for row, other in zip(raw["records"], seed2)
    ]
    baseline_phrase = _evidence_scores(
        raw["records"], baseline_risk, old_start, old_end, calibration
    )
    ensemble_phrase = _evidence_scores(
        raw["records"], ensemble_risk, averaged_start, averaged_end, calibration
    )
    baseline_task1 = task1_score(baseline_risk_f1, float(baseline_phrase.mean()))
    ensemble_task1 = task1_score(ensemble_risk_f1, float(ensemble_phrase.mean()))

    local_groups = groups[valid_idx]; unique = np.unique(local_groups)
    rng = np.random.default_rng(config.SEED + 1414); deltas = []
    for _ in range(3000):
        sampled_users = rng.choice(unique, size=len(unique), replace=True)
        sampled = np.concatenate([
            np.flatnonzero(local_groups == user) for user in sampled_users
        ])
        old_risk_score = f1_score(
            truth[sampled], baseline_risk[sampled], average="weighted", zero_division=0
        )
        new_risk_score = f1_score(
            truth[sampled], ensemble_risk[sampled], average="weighted", zero_division=0
        )
        deltas.append(
            task1_score(new_risk_score, float(ensemble_phrase[sampled].mean()))
            - task1_score(old_risk_score, float(baseline_phrase[sampled].mean()))
        )
    bootstrap = {
        "mean_delta": float(np.mean(deltas)),
        "p05_delta": float(np.quantile(deltas, 0.05)),
        "p95_delta": float(np.quantile(deltas, 0.95)),
        "positive_fraction": float(np.mean(np.asarray(deltas) > 0)),
    }
    adopted = bool(
        ensemble_task1 >= baseline_task1 + 0.003
        and bootstrap["positive_fraction"] >= 0.75
    )
    payload = {
        "training_version": TRAINING_VERSION,
        "seed2_history": seed2_payload["history"],
        "baseline": {
            "risk_f1": baseline_risk_f1,
            "phrase_f1": float(baseline_phrase.mean()), "task1": baseline_task1,
        },
        "fixed_equal_ensemble": {
            "risk_f1": ensemble_risk_f1,
            "phrase_f1": float(ensemble_phrase.mean()), "task1": ensemble_task1,
            "risk_confusion": confusion_matrix(truth, ensemble_risk, labels=np.arange(4)).tolist(),
            "improved_evidence_posts": int((ensemble_phrase > baseline_phrase).sum()),
            "worsened_evidence_posts": int((ensemble_phrase < baseline_phrase).sum()),
        },
        "user_cluster_bootstrap": bootstrap, "adopted": adopted,
    }
    RESULTS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    CALIBRATION.write_text(json.dumps({
        "training_version": TRAINING_VERSION, "adopted": adopted,
        "seed_weight": 0.5, "strict_baseline_task1": baseline_task1,
        "strict_ensemble_task1": ensemble_task1,
        "bootstrap_positive_fraction": bootstrap["positive_fraction"],
    }, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    return payload


if __name__ == "__main__":
    train_task1_seed_ensemble_v14()
