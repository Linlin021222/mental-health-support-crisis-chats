"""Definition-driven NLI risk expert, calibrated on user-disjoint folds."""
from __future__ import annotations

import json

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from analyze_task1_risk_v10 import _evidence_scores, _probabilities
from configs.config import config
from datasets.dataset import SuicideRiskDataset
from inference.factor_nli import _entailment_index, _word_chunks
from inference.task1_evidence_v4 import correct_risk_only, load_evidence_calibration
from utils.task1_metric import task1_score


OUTPUT = config.OUTPUT_DIR / "task1_nli_v15"
CACHE = OUTPUT / "strict_probabilities.npz"
RESULTS = OUTPUT / "results.json"
CALIBRATION = OUTPUT / "calibration.json"
HYPOTHESES = (
    "The author does not express suicidal thoughts, a suicide plan, or a previous suicide attempt.",
    "The author explicitly expresses suicidal thoughts or a wish to die, without describing a suicide method, plan, or previous attempt.",
    "The author expresses suicide and describes a current plan, method, preparation, or access to suicide means, but not a previous attempt.",
    "The author explicitly describes a past or recent suicide attempt that they carried out or survived.",
)


@torch.no_grad()
def _nli_probabilities(records):
    row_ids = np.asarray([str(row["row_id"]) for row in records])
    if CACHE.exists():
        saved = np.load(CACHE)
        if np.array_equal(saved["row_ids"].astype(str), row_ids):
            print("task1-nli-v15: resumed cached strict NLI scores", flush=True)
            return saved["probabilities"]
    device = torch.device(config.DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(
        config.FACTOR_NLI_MODEL_NAME, use_fast=True, local_files_only=True
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        config.FACTOR_NLI_MODEL_NAME, dtype=torch.float32, local_files_only=True
    ).to(device)
    model.eval(); entailment = _entailment_index(model)
    chunks = [_word_chunks(row["text"]) for row in records]
    probability = np.zeros((len(records), 4), dtype=np.float32)
    for label_id, hypothesis in enumerate(tqdm(HYPOTHESES, desc="task1 NLI hypotheses")):
        owners, premises = [], []
        for owner, windows in enumerate(chunks):
            owners.extend([owner] * len(windows)); premises.extend(windows)
        for start in range(0, len(premises), 16):
            stop = min(start + 16, len(premises))
            encoded = tokenizer(
                premises[start:stop], [hypothesis] * (stop - start),
                padding=True, truncation="only_first", max_length=384,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(**encoded).logits.float()
            values = (
                torch.sigmoid(logits[:, 0]) if logits.shape[-1] == 1
                else torch.softmax(logits, -1)[:, entailment]
            ).cpu().numpy()
            for owner, value in zip(owners[start:stop], values):
                probability[owner, label_id] = max(probability[owner, label_id], float(value))
    del model; torch.cuda.empty_cache()
    # A positive-suicide contradiction is an additional Indicator signal.
    probability[:, 0] = np.maximum(
        probability[:, 0], 1.0 - probability[:, 1:].max(1)
    )
    np.savez_compressed(CACHE, row_ids=row_ids, probabilities=probability)
    return probability


def _normalise_nli(probability, temperature):
    values = np.log(np.asarray(probability).clip(1e-6, 1.0)) / float(temperature)
    values -= values.max(1, keepdims=True); values = np.exp(values)
    return values / values.sum(1, keepdims=True)


def _predict(records, stable, nli, parameters):
    expert = _normalise_nli(nli, parameters["temperature"])
    mixed = (1.0 - parameters["nli_weight"]) * stable + parameters["nli_weight"] * expert
    raw = mixed.argmax(1)
    return np.asarray([
        correct_risk_only(row["text"], int(risk))
        for row, risk in zip(records, raw)
    ], dtype=np.int64)


def _search(records, truth, stable, nli, evidence_matrix, indices):
    subset = np.asarray(indices, dtype=int); rows = []
    for temperature in (0.35, 0.50, 0.75, 1.0, 1.5):
        for weight in (0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40):
            parameters = {"temperature": temperature, "nli_weight": weight}
            prediction = _predict(records, stable, nli, parameters)
            phrase = evidence_matrix[np.arange(len(records)), prediction]
            risk = float(f1_score(
                truth[subset], prediction[subset], average="weighted", zero_division=0
            ))
            phrase_f1 = float(phrase[subset].mean())
            rows.append({
                **parameters, "risk_f1": risk, "phrase_f1": phrase_f1,
                "task1": task1_score(risk, phrase_f1),
            })
    return sorted(rows, key=lambda row: row["task1"], reverse=True)


def _median_member(values, choices):
    median = float(np.median(values))
    return float(min(choices, key=lambda value: (abs(value - median), value)))


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
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
    stable = _probabilities(dataset, train_idx, valid_idx, raw["records"])
    nli = _nli_probabilities(raw["records"])
    evidence_calibration = load_evidence_calibration()
    evidence_matrix = np.column_stack([
        _evidence_scores(
            raw["records"], np.full(len(raw["records"]), risk_id), evidence_calibration
        ) for risk_id in range(4)
    ])
    truth = labels[valid_idx]; local_groups = groups[valid_idx]
    indices = np.arange(len(valid_idx))
    baseline_prediction = _predict(
        raw["records"], stable, nli, {"temperature": 1.0, "nli_weight": 0.0}
    )
    baseline_phrase = evidence_matrix[np.arange(len(indices)), baseline_prediction]
    baseline_risk = float(f1_score(
        truth, baseline_prediction, average="weighted", zero_division=0
    ))
    baseline_task1 = task1_score(baseline_risk, float(baseline_phrase.mean()))
    crossfit_prediction = np.empty(len(indices), dtype=np.int64)
    crossfit_phrase = np.empty(len(indices), dtype=np.float32); folds = []
    for fold, (fit, held) in enumerate(GroupKFold(n_splits=4).split(
        indices, groups=local_groups
    )):
        selected = _search(raw["records"], truth, stable, nli, evidence_matrix, fit)[0]
        parameters = {key: selected[key] for key in ("temperature", "nli_weight")}
        prediction = _predict(raw["records"], stable, nli, parameters)
        phrase = evidence_matrix[np.arange(len(indices)), prediction]
        crossfit_prediction[held] = prediction[held]; crossfit_phrase[held] = phrase[held]
        held_risk = float(f1_score(
            truth[held], prediction[held], average="weighted", zero_division=0
        ))
        folds.append({
            "fold": fold, **parameters, "heldout_risk_f1": held_risk,
            "heldout_phrase_f1": float(phrase[held].mean()),
            "heldout_task1": task1_score(held_risk, float(phrase[held].mean())),
        })
        print(f"task1-nli-v15 fold {fold + 1}/4 task1={folds[-1]['heldout_task1']:.4f}", flush=True)
    production = {
        "temperature": _median_member(
            [row["temperature"] for row in folds], (0.35, 0.50, 0.75, 1.0, 1.5)
        ),
        "nli_weight": _median_member(
            [row["nli_weight"] for row in folds], (0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40)
        ),
    }
    fixed_prediction = _predict(raw["records"], stable, nli, production)
    fixed_phrase = evidence_matrix[np.arange(len(indices)), fixed_prediction]
    fixed_risk = float(f1_score(
        truth, fixed_prediction, average="weighted", zero_division=0
    ))
    fixed_task1 = task1_score(fixed_risk, float(fixed_phrase.mean()))
    crossfit_risk = float(f1_score(
        truth, crossfit_prediction, average="weighted", zero_division=0
    ))
    crossfit_task1 = task1_score(crossfit_risk, float(crossfit_phrase.mean()))
    optimistic = _search(raw["records"], truth, stable, nli, evidence_matrix, indices)[0]
    rng = np.random.default_rng(config.SEED + 1515); unique = np.unique(local_groups)
    deltas = []
    for _ in range(3000):
        sampled_users = rng.choice(unique, size=len(unique), replace=True)
        sampled = np.concatenate([
            np.flatnonzero(local_groups == user) for user in sampled_users
        ])
        old_risk = f1_score(
            truth[sampled], baseline_prediction[sampled], average="weighted", zero_division=0
        )
        new_risk = f1_score(
            truth[sampled], fixed_prediction[sampled], average="weighted", zero_division=0
        )
        deltas.append(
            task1_score(new_risk, float(fixed_phrase[sampled].mean()))
            - task1_score(old_risk, float(baseline_phrase[sampled].mean()))
        )
    bootstrap = {
        "mean_delta": float(np.mean(deltas)), "p05_delta": float(np.quantile(deltas, 0.05)),
        "p95_delta": float(np.quantile(deltas, 0.95)),
        "positive_fraction": float(np.mean(np.asarray(deltas) > 0)),
    }
    adopted = bool(
        production["nli_weight"] > 0 and crossfit_task1 >= baseline_task1 + 0.003
        and fixed_task1 >= baseline_task1 + 0.003
        and bootstrap["positive_fraction"] >= 0.75
    )
    payload = {
        "training_version": "task1-definition-nli-v15",
        "baseline": {"risk_f1": baseline_risk, "phrase_f1": float(baseline_phrase.mean()), "task1": baseline_task1},
        "nli_standalone": {
            "risk_f1": float(f1_score(truth, nli.argmax(1), average="weighted", zero_division=0)),
            "confusion": confusion_matrix(truth, nli.argmax(1), labels=np.arange(4)).tolist(),
        },
        "nested_crossfit": {
            "risk_f1": crossfit_risk, "phrase_f1": float(crossfit_phrase.mean()),
            "task1": crossfit_task1, "folds": folds,
        },
        "fixed_production": {
            **production, "risk_f1": fixed_risk,
            "phrase_f1": float(fixed_phrase.mean()), "task1": fixed_task1,
            "confusion": confusion_matrix(truth, fixed_prediction, labels=np.arange(4)).tolist(),
        },
        "optimistic_full_holdout": optimistic,
        "user_cluster_bootstrap": bootstrap, "adopted": adopted,
    }
    RESULTS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    CALIBRATION.write_text(json.dumps({
        "training_version": payload["training_version"], "adopted": adopted,
        **production, "strict_fixed_task1": fixed_task1,
        "bootstrap_positive_fraction": bootstrap["positive_fraction"],
    }, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    return payload


if __name__ == "__main__":
    main()
