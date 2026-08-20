"""Dedicated ordinal risk model with leak-free lexical calibration (V27)."""
from __future__ import annotations

import copy
import json
import math

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import AutoModel, get_cosine_schedule_with_warmup

from analyze_task1_lexical_v11 import _lexical_experts, _softmax, _transformer_probability
from analyze_task1_atomic_refine_v26 import _refine_one
from baseline import _post_phrase_f1
from configs.config import config
from datasets.collator import SuicideRiskCollator
from datasets.dataset import SuicideRiskDataset
from inference.task1_evidence_v4 import apply_evidence_policy, correct_risk_only, decode_model_evidence
from preprocess.preprocess import load_train_data
from trainer.task1_atomic_v25 import _atomic_candidates, _bootstrap, _load_records
from utils.seed import seed_everything
from utils.task1_metric import task1_score


OUTPUT = config.OUTPUT_DIR / "task1_risk_only_v27"
RESULTS = OUTPUT / "results.json"
CALIBRATION = OUTPUT / "calibration.json"
INNER_CHECKPOINT = OUTPUT / "inner_model.pt"
OUTER_CHECKPOINT = OUTPUT / "outer_model.pt"
TRAINING_VERSION = "task1-dedicated-ordinal-risk-v27"
INNER_CAL_FOLD = 3
MAX_EPOCHS = 4
ACCUMULATION = 8


class RiskOnlyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(
            config.MODEL_NAME, local_files_only=True, dtype=torch.float32
        )
        self.encoder.gradient_checkpointing_disable()
        hidden = int(self.encoder.config.hidden_size)
        self.chunk_attention = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.Tanh(), nn.Linear(hidden // 2, 1)
        )
        self.dropout = nn.Dropout(0.15)
        self.classifier = nn.Linear(hidden, 4)
        self.ordinal = nn.Linear(hidden, 3)

    def forward(self, input_ids, attention_mask):
        batch, chunks, length = input_ids.shape
        flat_ids = input_ids.reshape(batch * chunks, length)
        flat_mask = attention_mask.reshape(batch * chunks, length)
        hidden = self.encoder(
            input_ids=flat_ids, attention_mask=flat_mask
        ).last_hidden_state
        weights = flat_mask.to(hidden.dtype).unsqueeze(-1)
        pooled = (hidden * weights).sum(1) / weights.sum(1).clamp_min(1.0)
        pooled = pooled.reshape(batch, chunks, -1)
        valid_chunks = attention_mask.any(-1)
        scores = self.chunk_attention(pooled).squeeze(-1).masked_fill(
            ~valid_chunks, -1e4
        )
        document = (pooled * torch.softmax(scores, -1).unsqueeze(-1)).sum(1)
        document = self.dropout(document)
        return self.classifier(document), self.ordinal(document)


def _loader(dataset, indices, shuffle):
    return DataLoader(
        Subset(dataset, list(map(int, indices))), batch_size=1, shuffle=shuffle,
        collate_fn=SuicideRiskCollator(), num_workers=0, pin_memory=True,
    )


def _weights(labels, indices, device):
    counts = np.bincount(labels[np.asarray(indices)], minlength=4).astype(float)
    class_weights = np.sqrt(counts.sum() / np.maximum(counts, 1.0))
    class_weights /= class_weights.mean()
    ordinal_targets = labels[np.asarray(indices), None] > np.arange(3)[None, :]
    positive = ordinal_targets.sum(0).astype(float)
    negative = len(indices) - positive
    ordinal_weights = np.sqrt(negative / np.maximum(positive, 1.0))
    return (
        torch.tensor(class_weights, dtype=torch.float32, device=device),
        torch.tensor(ordinal_weights, dtype=torch.float32, device=device),
    )


def _probability(logits, ordinal_logits):
    standard = torch.softmax(logits, -1).cpu().numpy()
    cumulative = torch.sigmoid(ordinal_logits).cpu().numpy()
    cumulative = np.minimum.accumulate(cumulative, axis=1)
    ordinal = np.column_stack([
        1.0 - cumulative[:, 0],
        cumulative[:, 0] - cumulative[:, 1],
        cumulative[:, 1] - cumulative[:, 2],
        cumulative[:, 2],
    ]).clip(0.0, 1.0)
    ordinal /= ordinal.sum(1, keepdims=True).clip(1e-8)
    return 0.75 * standard + 0.25 * ordinal


@torch.no_grad()
def _infer(model, dataset, indices, device, description):
    model.eval(); result = []
    for batch in tqdm(_loader(dataset, indices, False), desc=description):
        logits, ordinal = model(
            batch["input_ids"].to(device), batch["attention_mask"].to(device)
        )
        result.append(_probability(logits, ordinal)[0])
    return np.vstack(result)


def _train(model, dataset, indices, labels, device, epochs, callback=None):
    loader = _loader(dataset, indices, True)
    head_ids = {id(p) for module in (model.classifier, model.ordinal, model.chunk_attention)
                for p in module.parameters()}
    optimizer = AdamW([
        {"params": [p for p in model.parameters() if id(p) not in head_ids], "lr": 8e-6},
        {"params": [p for p in model.parameters() if id(p) in head_ids], "lr": 4e-5},
    ], weight_decay=0.01)
    updates = math.ceil(len(loader) / ACCUMULATION) * epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, max(1, int(0.1 * updates)), max(1, updates)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    class_weights, ordinal_weights = _weights(labels, indices, device)
    history = []
    for epoch in range(1, epochs + 1):
        model.train(); optimizer.zero_grad(set_to_none=True); losses = []
        for step, batch in enumerate(tqdm(loader, desc=f"V27 epoch {epoch}/{epochs}"), 1):
            target = batch["risk_labels"].to(device)
            ordinal_target = (target[:, None] > torch.arange(3, device=device)[None, :]).float()
            with torch.autocast(device_type="cuda", enabled=True):
                logits, ordinal = model(
                    batch["input_ids"].to(device), batch["attention_mask"].to(device)
                )
                ce = nn.functional.cross_entropy(
                    logits, target, weight=class_weights, label_smoothing=0.04
                )
                ordinal_loss = nn.functional.binary_cross_entropy_with_logits(
                    ordinal, ordinal_target, pos_weight=ordinal_weights
                )
                loss = (ce + 0.35 * ordinal_loss) / ACCUMULATION
            scaler.scale(loss).backward(); losses.append(float(loss.detach()) * ACCUMULATION)
            if step % ACCUMULATION == 0 or step == len(loader):
                scaler.unscale_(optimizer); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                old_scale = scaler.get_scale()
                scaler.step(optimizer); scaler.update()
                if scaler.get_scale() >= old_scale:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses))}
        history.append(row); print(f"V27 epoch={epoch} loss={row['train_loss']:.4f}", flush=True)
        if callback:
            callback(epoch, model, row)
    return history


def _calibrated_prediction(texts, old_probability, new_probability, lexical_decision, parameters):
    lexical = _softmax(lexical_decision, parameters["temperature"])
    neural = ((1.0 - parameters["new_weight"]) * old_probability
              + parameters["new_weight"] * new_probability)
    probability = ((1.0 - parameters["lexical_weight"]) * neural
                   + parameters["lexical_weight"] * lexical)
    logits = np.log(probability.clip(1e-8, 1.0))
    logits[:, 0] += parameters["indicator_bias"]
    logits[:, 2] += parameters["behavior_bias"]
    logits[:, 3] += parameters["attempt_bias"]
    raw = logits.argmax(1)
    return np.asarray([
        correct_risk_only(text, int(risk)) for text, risk in zip(texts, raw)
    ], dtype=np.int64)


def _grid(texts, truth, old_probability, new_probability, lexical_decision):
    rows = []
    for new_weight in (0.25, 0.50, 0.75, 1.0):
        for lexical_weight in (0.30, 0.45, 0.60, 0.75):
            for temperature in (0.50, 0.75, 1.0):
                for indicator_bias in (-0.10, 0.0, 0.10):
                    for behavior_bias in (-0.15, 0.0, 0.15):
                        for attempt_bias in (0.0, 0.20, 0.40):
                            parameters = {
                                "new_weight": new_weight, "lexical_weight": lexical_weight,
                                "temperature": temperature, "indicator_bias": indicator_bias,
                                "behavior_bias": behavior_bias, "attempt_bias": attempt_bias,
                            }
                            prediction = _calibrated_prediction(
                                texts, old_probability, new_probability,
                                lexical_decision, parameters,
                            )
                            rows.append({
                                **parameters,
                                "risk_f1": float(f1_score(
                                    truth, prediction, average="weighted", zero_division=0
                                )),
                            })
    return sorted(rows, key=lambda row: row["risk_f1"], reverse=True)


def _v18_evidence(records, seed2, risks, evidence_parameters):
    result = []
    for record, second, risk in zip(records, seed2, risks):
        parameters = evidence_parameters[config.ID2RISK[int(risk)]]
        start = 0.8 * record["start"] + 0.2 * second["start"]
        end = 0.8 * record["end"] + 0.2 * second["end"]
        spans = decode_model_evidence(
            record["text"], record["offsets"], start, end,
            parameters["threshold"], parameters["max_tokens"],
            parameters["end_policy"], limit=5,
        )
        result.append(apply_evidence_policy(
            record["text"], int(risk), spans,
            parameters["cue_policy"], parameters["topk"],
        ))
    return result


def train_task1_risk_only_v27():
    if not torch.cuda.is_available():
        raise RuntimeError("V27 requires CUDA")
    OUTPUT.mkdir(parents=True, exist_ok=True); device = torch.device("cuda")
    seed_everything(config.SEED + 2727)
    frame = load_train_data().reset_index(drop=True)
    dataset = SuicideRiskDataset(config.CACHE_DIR / "train_cache.pt")
    labels = frame.risk_label.to_numpy(); groups = frame.anon_user_id.astype(str).to_numpy()
    outer_train, outer_valid = next(StratifiedGroupKFold(
        n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED
    ).split(np.zeros(len(frame)), labels, groups))
    inner_records, membership, outer_raw = _load_records()
    inner_fit = np.asarray([i for i in outer_train if membership[int(i)] != INNER_CAL_FOLD])
    inner_cal = np.asarray([i for i in outer_train if membership[int(i)] == INNER_CAL_FOLD])
    inner_by_index = {int(row["global_index"]): row for row in inner_records}
    old_inner = np.vstack([inner_by_index[int(i)]["old_probability"] for i in inner_cal])
    inner_texts = frame.text.iloc[inner_cal].astype(str).tolist()
    inner_lexical = _lexical_experts(frame, inner_fit, inner_cal)["svc-c0.25-balanced"]
    model = RiskOnlyModel().to(device); selected = {"risk_f1": -1.0}; state = None; top = []

    def callback(epoch, current, row):
        nonlocal selected, state, top
        probability = _infer(current, dataset, inner_cal, device, f"V27 inner cal epoch {epoch}")
        grid = _grid(inner_texts, labels[inner_cal], old_inner, probability, inner_lexical)
        row["calibration_risk_f1"] = grid[0]["risk_f1"]
        print(f"V27 epoch={epoch} inner risk_f1={grid[0]['risk_f1']:.6f}", flush=True)
        if grid[0]["risk_f1"] > selected["risk_f1"]:
            selected = {"epoch": epoch, **grid[0]}; top = grid[:20]
            state = copy.deepcopy({k: v.detach().cpu() for k, v in current.state_dict().items()})

    history = _train(model, dataset, inner_fit, labels, device, MAX_EPOCHS, callback)
    torch.save({"training_version": TRAINING_VERSION, "state_dict": state,
                "selected": selected, "history": history}, INNER_CHECKPOINT)
    del model, state; torch.cuda.empty_cache()

    seed_everything(config.SEED + 2728)
    final = RiskOnlyModel().to(device)
    refit_history = _train(final, dataset, outer_train, labels, device, int(selected["epoch"]))
    torch.save({"training_version": TRAINING_VERSION, "state_dict": final.state_dict(),
                "selected": selected, "history": refit_history}, OUTER_CHECKPOINT)
    new_outer = _infer(final, dataset, outer_valid, device, "V27 untouched outer risk")
    outer_records = outer_raw["records"]
    old_outer = np.vstack([row["old_probability"] for row in outer_records])
    outer_lexical = _lexical_experts(frame, outer_train, outer_valid)["svc-c0.25-balanced"]
    candidate_risk = _calibrated_prediction(
        frame.text.iloc[outer_valid].astype(str).tolist(), old_outer, new_outer,
        outer_lexical, selected,
    )

    # Current production V18 is the comparator, not the weaker V4 baseline.
    transformer = _transformer_probability(dataset, outer_valid, outer_records)
    v18_parameters = json.loads((
        config.OUTPUT_DIR / "task1_candidate_v18" / "results.json"
    ).read_text(encoding="utf-8"))
    baseline_risk = _calibrated_prediction(
        frame.text.iloc[outer_valid].astype(str).tolist(), transformer, transformer,
        outer_lexical, {
            "new_weight": 0.0, "lexical_weight": 0.60, "temperature": 1.0,
            "indicator_bias": 0.0, "behavior_bias": 0.0, "attempt_bias": 0.20,
        },
    )
    seed2 = torch.load(
        config.OUTPUT_DIR / "task1_seed_ensemble_v14" / "seed2_valid.pt",
        map_location="cpu", weights_only=False,
    )["rows"]
    evidence_parameters = v18_parameters["evidence_parameters_by_predicted_risk"]
    baseline_evidence = _v18_evidence(outer_records, seed2, baseline_risk, evidence_parameters)
    candidate_evidence = _v18_evidence(outer_records, seed2, candidate_risk, evidence_parameters)

    # Apply the independently inner-selected V26 boundary policy if available.
    boundary_file = config.OUTPUT_DIR / "task1_atomic_refine_v26" / "results.json"
    atomic_file = config.OUTPUT_DIR / "task1_atomic_refine_v26" / "atomic_outputs.pt"
    if boundary_file.exists() and atomic_file.exists():
        boundary = json.loads(boundary_file.read_text(encoding="utf-8"))["selected"]
        atomic_outputs = torch.load(atomic_file, map_location="cpu", weights_only=False)["outer_outputs"]
        refined = []
        for global_index, evidence, risk in zip(outer_valid, candidate_evidence, candidate_risk):
            text = str(frame.iloc[int(global_index)].text)
            atomic = _atomic_candidates(
                text, atomic_outputs.get(int(global_index), []),
                boundary["token_threshold"], boundary["sentence_threshold"],
                boundary["max_tokens"],
            )
            refined.append([] if int(risk) == 0 else _refine_one(text, evidence, atomic, boundary))
        candidate_evidence = refined

    gold = [list(frame.iloc[int(i)].evidence) for i in outer_valid]
    baseline_phrase = np.asarray([_post_phrase_f1(x, y) for x, y in zip(baseline_evidence, gold)])
    candidate_phrase = np.asarray([_post_phrase_f1(x, y) for x, y in zip(candidate_evidence, gold)])
    baseline_risk_f1 = float(f1_score(labels[outer_valid], baseline_risk, average="weighted"))
    candidate_risk_f1 = float(f1_score(labels[outer_valid], candidate_risk, average="weighted"))
    baseline_task = task1_score(baseline_risk_f1, baseline_phrase.mean())
    candidate_task = task1_score(candidate_risk_f1, candidate_phrase.mean())
    # Cluster bootstrap recomputes the non-linear weighted risk F1.
    unique = np.unique(groups[outer_valid]); rng = np.random.default_rng(config.SEED + 2727); deltas = []
    for _ in range(4000):
        sampled_users = rng.choice(unique, size=len(unique), replace=True)
        positions = np.concatenate([np.flatnonzero(groups[outer_valid] == u) for u in sampled_users])
        old_risk = f1_score(labels[outer_valid][positions], baseline_risk[positions],
                            average="weighted", zero_division=0)
        new_risk = f1_score(labels[outer_valid][positions], candidate_risk[positions],
                            average="weighted", zero_division=0)
        deltas.append(task1_score(new_risk, candidate_phrase[positions].mean())
                      - task1_score(old_risk, baseline_phrase[positions].mean()))
    bootstrap = {"mean_delta": float(np.mean(deltas)),
                 "p05_delta": float(np.quantile(deltas, .05)),
                 "p95_delta": float(np.quantile(deltas, .95)),
                 "positive_fraction": float((np.asarray(deltas) > 0).mean())}
    adopted = bool(candidate_task >= baseline_task + .003 and bootstrap["positive_fraction"] >= .80)
    payload = {
        "training_version": TRAINING_VERSION,
        "evaluation_scope": "epoch/calibration on inner users; refit outer train; one outer evaluation",
        "selected": selected, "inner_top20": top, "history": history,
        "refit_history": refit_history,
        "baseline_v18": {"risk_f1": baseline_risk_f1, "phrase_f1": float(baseline_phrase.mean()),
                         "task1": baseline_task, "confusion": confusion_matrix(labels[outer_valid], baseline_risk).tolist()},
        "candidate": {"risk_f1": candidate_risk_f1, "phrase_f1": float(candidate_phrase.mean()),
                      "task1": candidate_task, "confusion": confusion_matrix(labels[outer_valid], candidate_risk).tolist()},
        "user_cluster_bootstrap": bootstrap, "adopted": adopted,
    }
    RESULTS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    CALIBRATION.write_text(json.dumps({"training_version": TRAINING_VERSION, "adopted": adopted,
        "strict_task1": candidate_task, "selected": selected}, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True); return payload


if __name__ == "__main__":
    train_task1_risk_only_v27()
