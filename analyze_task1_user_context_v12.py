"""Leak-free within-split user-history aggregation for Task 1 risk."""
from __future__ import annotations

import json

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

from analyze_task1_risk_v10 import _evidence_scores, _probabilities
from configs.config import config
from datasets.dataset import SuicideRiskDataset
from inference.task1_evidence_v4 import correct_risk_only, load_evidence_calibration
from utils.task1_metric import task1_score


OUTPUT = config.OUTPUT_DIR / "task1_user_context_v12"
RESULTS = OUTPUT / "results.json"
CALIBRATION = OUTPUT / "calibration.json"
TRAINING_VERSION = "task1-user-context-v12"


def _user_prior(probability, groups, mode):
    prior = np.empty_like(probability, dtype=np.float64)
    for user in np.unique(groups):
        indices = np.flatnonzero(groups == user)
        block = probability[indices]
        if mode == "arithmetic":
            summary = block.mean(0)
        elif mode == "geometric":
            logits = np.log(block.clip(1e-8, 1.0)).mean(0)
            logits -= logits.max(); summary = np.exp(logits)
            summary /= summary.sum()
        elif mode == "trimmed":
            # Median is robust when one user has one unusually severe post.
            summary = np.median(block, axis=0)
            summary /= summary.sum()
        else:
            raise ValueError(mode)
        prior[indices] = summary
    return prior


def _predict(records, probability, prior, parameters):
    alpha = float(parameters["context_weight"])
    combined = (1.0 - alpha) * probability + alpha * prior
    logits = np.log(combined.clip(1e-8, 1.0))
    logits[:, config.RISK_LABELS["Attempt"]] += float(parameters["attempt_bias"])
    raw = logits.argmax(1)
    return np.asarray([
        correct_risk_only(row["text"], int(risk))
        for row, risk in zip(records, raw)
    ], dtype=np.int64)


def _search(records, truth, probability, priors, evidence_matrix, indices):
    subset = np.asarray(indices, dtype=int); rows = []
    for mode, prior in priors.items():
        for weight in (0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60):
            for attempt_bias in (0.0, 0.20, 0.40):
                parameters = {
                    "context_mode": mode, "context_weight": weight,
                    "attempt_bias": attempt_bias,
                }
                prediction = _predict(records, probability, prior, parameters)
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
        raise ValueError("V12 and V4 strict folds differ")
    records = raw["records"]
    probability = _probabilities(dataset, train_idx, valid_idx, records)
    local_groups = groups[valid_idx]
    priors = {
        mode: _user_prior(probability, local_groups, mode)
        for mode in ("arithmetic", "geometric", "trimmed")
    }
    calibration = load_evidence_calibration()
    if calibration is None:
        raise FileNotFoundError("Adopted evidence calibration is required")
    print("user-context-v12: caching evidence outcomes...", flush=True)
    evidence_matrix = np.column_stack([
        _evidence_scores(
            records, np.full(len(records), risk_id, dtype=np.int64), calibration
        )
        for risk_id in range(config.NUM_RISK_CLASSES)
    ])
    truth = labels[valid_idx]; indices = np.arange(len(records))
    baseline_parameters = {
        "context_mode": "arithmetic", "context_weight": 0.0,
        "attempt_bias": 0.0,
    }
    baseline_prediction = _predict(
        records, probability, priors["arithmetic"], baseline_parameters
    )
    baseline_phrase = evidence_matrix[np.arange(len(records)), baseline_prediction]
    baseline_risk = float(f1_score(
        truth, baseline_prediction, average="weighted", zero_division=0
    ))
    baseline_task1 = task1_score(baseline_risk, float(baseline_phrase.mean()))

    print("user-context-v12: nested user-level context search...", flush=True)
    crossfit_prediction = np.empty(len(records), dtype=np.int64)
    crossfit_phrase = np.empty(len(records), dtype=np.float32); selections = []
    for fold, (fit, held) in enumerate(GroupKFold(n_splits=4).split(
        indices, groups=local_groups
    )):
        selected = _search(
            records, truth, probability, priors, evidence_matrix, fit
        )[0]
        parameters = {key: selected[key] for key in (
            "context_mode", "context_weight", "attempt_bias"
        )}
        prediction = _predict(
            records, probability, priors[parameters["context_mode"]], parameters
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
            f"user-context-v12: fold {fold + 1}/4 "
            f"task1={selections[-1]['heldout_task1']:.4f}", flush=True,
        )
    production = {
        "context_mode": _mode([row["context_mode"] for row in selections]),
        "context_weight": float(_median_member([
            row["context_weight"] for row in selections
        ])),
        "attempt_bias": float(_median_member([
            row["attempt_bias"] for row in selections
        ])),
    }
    fixed_prediction = _predict(
        records, probability, priors[production["context_mode"]], production
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
        records, truth, probability, priors, evidence_matrix, indices
    )[0]

    rng = np.random.default_rng(config.SEED + 1212)
    unique = np.unique(local_groups); deltas = []
    for _ in range(3000):
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
        production["context_weight"] > 0
        and crossfit_task1 >= baseline_task1 + 0.003
        and fixed_task1 >= baseline_task1 + 0.003
        and bootstrap["positive_fraction"] >= 0.75
    )
    payload = {
        "training_version": TRAINING_VERSION,
        "strict_users": int(len(np.unique(local_groups))),
        "mean_posts_per_user": float(len(records) / len(np.unique(local_groups))),
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
