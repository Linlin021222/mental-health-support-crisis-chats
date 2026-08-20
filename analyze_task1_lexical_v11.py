"""Nested evaluation of discriminative and NB-SVM lexical risk experts."""
from __future__ import annotations

import json

import numpy as np
import torch
from scipy import sparse
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.svm import LinearSVC

from analyze_task1_v2_ensemble import collect
from analyze_task1_risk_v10 import _evidence_scores
from baseline import _vectorizer
from configs.config import config
from datasets.dataset import SuicideRiskDataset
from inference.task1_evidence_v4 import correct_risk_only, load_evidence_calibration
from models.multitask_model_v2 import SuicideRiskMultiTaskModelV2
from preprocess.preprocess import load_train_data
from utils.task1_metric import task1_score


OUTPUT = config.OUTPUT_DIR / "task1_lexical_v11"
RESULTS = OUTPUT / "results.json"
CALIBRATION = OUTPUT / "calibration.json"
TRAINING_VERSION = "task1-nbsvm-lexical-v11"


def _softmax(values, temperature):
    values = np.asarray(values, dtype=np.float64) / float(temperature)
    values -= values.max(1, keepdims=True)
    values = np.exp(values)
    return values / values.sum(1, keepdims=True)


def _transformer_probability(dataset, valid_idx, records):
    print("lexical-v11: collecting ordinal Transformer expert...", flush=True)
    device = torch.device(config.DEVICE)
    model = SuicideRiskMultiTaskModelV2().to(device)
    model.load_state_dict(torch.load(
        config.OUTPUT_DIR / "task1_v2_strict_model.pt", map_location=device
    ))
    v2 = collect(model, dataset, valid_idx, device, v2=True)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    old = np.vstack([row["old_probability"] for row in records])
    v2_probability = np.vstack([
        0.75 * row["standard"] + 0.25 * row["ordinal"] for row in v2
    ])
    return 0.8 * old + 0.2 * v2_probability


def _nbsvm_decision(train_matrix, train_labels, valid_matrix, c_value):
    decisions = []
    binary = train_matrix.copy()
    binary.data = np.ones_like(binary.data)
    for class_id in range(config.NUM_RISK_CLASSES):
        target = (train_labels == class_id).astype(np.int64)
        positive = np.asarray(binary[target == 1].sum(0)).ravel() + 1.0
        negative = np.asarray(binary[target == 0].sum(0)).ravel() + 1.0
        ratio = np.log(positive / positive.sum()) - np.log(negative / negative.sum())
        transformed_train = train_matrix.multiply(ratio)
        classifier = LinearSVC(C=float(c_value), class_weight="balanced")
        classifier.fit(transformed_train, target)
        decisions.append(classifier.decision_function(valid_matrix.multiply(ratio)))
    return np.column_stack(decisions)


def _lexical_experts(frame, train_idx, valid_idx):
    print("lexical-v11: fitting word/character experts...", flush=True)
    vectorizer = _vectorizer()
    train_matrix = vectorizer.fit_transform(frame.text.iloc[train_idx])
    valid_matrix = vectorizer.transform(frame.text.iloc[valid_idx])
    labels = np.asarray(frame.risk_label)[train_idx]
    experts = {}
    for c_value in (0.25, 0.50, 1.0, 2.0):
        for class_weight in (None, "balanced"):
            name = f"svc-c{c_value:g}-{'balanced' if class_weight else 'plain'}"
            classifier = LinearSVC(C=c_value, class_weight=class_weight).fit(
                train_matrix, labels
            )
            experts[name] = classifier.decision_function(valid_matrix)
    for c_value in (0.25, 0.50, 1.0):
        experts[f"nbsvm-c{c_value:g}"] = _nbsvm_decision(
            train_matrix, labels, valid_matrix, c_value
        )
    return experts


def _prediction(records, transformer, decision, parameters):
    lexical = _softmax(decision, parameters["temperature"])
    probability = (
        (1.0 - parameters["lexical_weight"]) * transformer
        + parameters["lexical_weight"] * lexical
    )
    logits = np.log(probability.clip(1e-8, 1.0))
    logits[:, config.RISK_LABELS["Attempt"]] += parameters["attempt_bias"]
    raw = logits.argmax(1)
    return np.asarray([
        correct_risk_only(row["text"], int(risk))
        for row, risk in zip(records, raw)
    ], dtype=np.int64)


def _search(records, truth, transformer, experts, evidence_matrix, indices):
    subset = np.asarray(indices, dtype=int); rows = []
    for expert_name, decision in experts.items():
        for temperature in (0.30, 0.50, 0.70, 1.0):
            for lexical_weight in (0.15, 0.30, 0.45, 0.60):
                for attempt_bias in (0.0, 0.20, 0.40):
                    parameters = {
                        "expert": expert_name, "temperature": temperature,
                        "lexical_weight": lexical_weight,
                        "attempt_bias": attempt_bias,
                    }
                    prediction = _prediction(
                        records, transformer, decision, parameters
                    )
                    phrase = evidence_matrix[np.arange(len(records)), prediction]
                    risk = float(f1_score(
                        truth[subset], prediction[subset],
                        average="weighted", zero_division=0,
                    ))
                    phrase_f1 = float(phrase[subset].mean())
                    rows.append({
                        **parameters, "risk_f1": risk, "phrase_f1": phrase_f1,
                        "task1": task1_score(risk, phrase_f1),
                    })
    return sorted(rows, key=lambda row: row["task1"], reverse=True)


def _mode(values):
    from collections import Counter
    return Counter(values).most_common(1)[0][0]


def _median_member(values):
    median = float(np.median(values))
    return min(values, key=lambda value: (abs(float(value) - median), float(value)))


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
    if not np.array_equal(np.asarray(raw["valid_idx"]), valid_idx):
        raise ValueError("V11 and V4 strict folds differ")
    records = raw["records"]
    frame = load_train_data().reset_index(drop=True)
    transformer = _transformer_probability(dataset, valid_idx, records)
    experts = _lexical_experts(frame, train_idx, valid_idx)
    calibration = load_evidence_calibration()
    if calibration is None:
        raise FileNotFoundError("Adopted evidence calibration is required")
    print("lexical-v11: caching evidence outcomes...", flush=True)
    evidence_matrix = np.column_stack([
        _evidence_scores(
            records, np.full(len(records), risk_id, dtype=np.int64), calibration
        )
        for risk_id in range(config.NUM_RISK_CLASSES)
    ])
    truth = labels[valid_idx]; local_groups = groups[valid_idx]
    indices = np.arange(len(records))
    baseline_parameters = {
        "temperature": 0.50, "lexical_weight": 0.30,
        "attempt_bias": 0.0,
    }
    baseline_decision = experts["svc-c1-balanced"]
    baseline_prediction = _prediction(
        records, transformer, baseline_decision,
        {"expert": "svc-c1-balanced", **baseline_parameters},
    )
    baseline_phrase = evidence_matrix[np.arange(len(records)), baseline_prediction]
    baseline_risk = float(f1_score(
        truth, baseline_prediction, average="weighted", zero_division=0
    ))
    baseline_task1 = task1_score(baseline_risk, float(baseline_phrase.mean()))

    print("lexical-v11: nested user-level expert selection...", flush=True)
    crossfit_prediction = np.empty(len(records), dtype=np.int64)
    crossfit_phrase = np.empty(len(records), dtype=np.float32); selections = []
    for fold, (fit, held) in enumerate(GroupKFold(n_splits=4).split(
        indices, groups=local_groups
    )):
        selected = _search(
            records, truth, transformer, experts, evidence_matrix, fit
        )[0]
        parameters = {key: selected[key] for key in (
            "expert", "temperature", "lexical_weight", "attempt_bias"
        )}
        prediction = _prediction(
            records, transformer, experts[parameters["expert"]], parameters
        )
        phrase = evidence_matrix[np.arange(len(records)), prediction]
        crossfit_prediction[held] = prediction[held]
        crossfit_phrase[held] = phrase[held]
        held_risk = float(f1_score(
            truth[held], prediction[held], average="weighted", zero_division=0
        ))
        selections.append({
            "fold": fold, **parameters,
            "heldout_risk_f1": held_risk,
            "heldout_phrase_f1": float(phrase[held].mean()),
            "heldout_task1": task1_score(held_risk, float(phrase[held].mean())),
        })
        print(
            f"lexical-v11: fold {fold + 1}/4 "
            f"task1={selections[-1]['heldout_task1']:.4f}", flush=True,
        )
    production = {
        "expert": _mode([row["expert"] for row in selections]),
        "temperature": float(_median_member([row["temperature"] for row in selections])),
        "lexical_weight": float(_median_member([row["lexical_weight"] for row in selections])),
        "attempt_bias": float(_median_member([row["attempt_bias"] for row in selections])),
    }
    fixed_prediction = _prediction(
        records, transformer, experts[production["expert"]], production
    )
    fixed_phrase = evidence_matrix[np.arange(len(records)), fixed_prediction]
    fixed_risk = float(f1_score(
        truth, fixed_prediction, average="weighted", zero_division=0
    ))
    crossfit_risk = float(f1_score(
        truth, crossfit_prediction, average="weighted", zero_division=0
    ))
    crossfit_task1 = task1_score(crossfit_risk, float(crossfit_phrase.mean()))
    fixed_task1 = task1_score(fixed_risk, float(fixed_phrase.mean()))
    optimistic = _search(
        records, truth, transformer, experts, evidence_matrix, indices
    )[0]

    rng = np.random.default_rng(config.SEED + 1111)
    unique = np.unique(local_groups); deltas = []
    for _ in range(2000):
        sampled_users = rng.choice(unique, size=len(unique), replace=True)
        sampled = np.concatenate([
            np.flatnonzero(local_groups == user) for user in sampled_users
        ])
        old_risk = f1_score(
            truth[sampled], baseline_prediction[sampled],
            average="weighted", zero_division=0,
        )
        new_risk = f1_score(
            truth[sampled], fixed_prediction[sampled],
            average="weighted", zero_division=0,
        )
        deltas.append(
            task1_score(new_risk, float(fixed_phrase[sampled].mean()))
            - task1_score(old_risk, float(baseline_phrase[sampled].mean()))
        )
    bootstrap = {
        "mean_delta": float(np.mean(deltas)),
        "p05_delta": float(np.quantile(deltas, 0.05)),
        "p95_delta": float(np.quantile(deltas, 0.95)),
        "positive_fraction": float(np.mean(np.asarray(deltas) > 0)),
    }
    adopted = bool(
        crossfit_task1 >= baseline_task1 + 0.003
        and fixed_task1 >= baseline_task1 + 0.003
        and bootstrap["positive_fraction"] >= 0.75
    )
    payload = {
        "training_version": TRAINING_VERSION,
        "baseline": {
            "risk_f1": baseline_risk, "phrase_f1": float(baseline_phrase.mean()),
            "task1": baseline_task1,
        },
        "nested_crossfit": {
            "risk_f1": crossfit_risk, "phrase_f1": float(crossfit_phrase.mean()),
            "task1": crossfit_task1, "folds": selections,
        },
        "fixed_production": {
            **production, "risk_f1": fixed_risk,
            "phrase_f1": float(fixed_phrase.mean()), "task1": fixed_task1,
            "confusion": confusion_matrix(truth, fixed_prediction, labels=np.arange(4)).tolist(),
        },
        "optimistic_full_holdout": optimistic,
        "user_cluster_bootstrap": bootstrap,
        "adopted": adopted,
    }
    RESULTS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    CALIBRATION.write_text(json.dumps({
        "training_version": TRAINING_VERSION, "adopted": adopted,
        **production, "strict_baseline_task1": baseline_task1,
        "strict_crossfit_task1": crossfit_task1, "strict_fixed_task1": fixed_task1,
        "bootstrap_positive_fraction": bootstrap["positive_fraction"],
    }, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    return payload


if __name__ == "__main__":
    main()
