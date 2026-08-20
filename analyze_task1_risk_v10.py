"""User-disjoint ordinal class-bias calibration for the stable Task 1 risk ensemble."""
from __future__ import annotations

import json
from collections import Counter

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.svm import LinearSVC

from analyze_task1_v2_ensemble import collect
from baseline import _post_phrase_f1, _vectorizer
from configs.config import config
from datasets.dataset import SuicideRiskDataset
from inference.task1_evidence_v4 import (
    apply_evidence_policy, correct_risk_only, decode_model_evidence,
    load_evidence_calibration,
)
from models.multitask_model_v2 import SuicideRiskMultiTaskModelV2
from preprocess.preprocess import load_train_data
from utils.task1_metric import task1_score


OUTPUT = config.OUTPUT_DIR / "task1_risk_v10"
RESULTS = OUTPUT / "results.json"
CALIBRATION = OUTPUT / "calibration.json"
TRAINING_VERSION = "task1-ordinal-bias-v10"


def _softmax(values, temperature=0.5):
    values = np.asarray(values, dtype=np.float64) / float(temperature)
    values -= values.max(1, keepdims=True)
    values = np.exp(values)
    return values / values.sum(1, keepdims=True)


def _probabilities(dataset, train_idx, valid_idx, records):
    device = torch.device(config.DEVICE)
    print("risk-v10: collecting ordinal expert probabilities...", flush=True)
    model = SuicideRiskMultiTaskModelV2().to(device)
    model.load_state_dict(torch.load(
        config.OUTPUT_DIR / "task1_v2_strict_model.pt", map_location=device
    ))
    v2_records = collect(model, dataset, valid_idx, device, v2=True)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    frame = load_train_data().reset_index(drop=True)
    vectorizer = _vectorizer()
    train_matrix = vectorizer.fit_transform(frame.text.iloc[train_idx])
    valid_matrix = vectorizer.transform(frame.text.iloc[valid_idx])
    lexical_model = LinearSVC(C=1.0, class_weight="balanced").fit(
        train_matrix, np.asarray(frame.risk_label)[train_idx]
    )
    lexical = _softmax(lexical_model.decision_function(valid_matrix), 0.5)
    old = np.vstack([row["old_probability"] for row in records])
    v2 = np.vstack([
        0.75 * row["standard"] + 0.25 * row["ordinal"] for row in v2_records
    ])
    transformer = 0.8 * old + 0.2 * v2
    return 0.7 * transformer + 0.3 * lexical


def _predict(records, probability, parameters):
    log_probability = np.log(np.asarray(probability).clip(1e-8, 1.0))
    bias = np.asarray([
        parameters["indicator_bias"], 0.0,
        parameters["behavior_bias"], parameters["attempt_bias"],
    ])
    raw = np.argmax(log_probability + bias[None, :], axis=1)
    return np.asarray([
        correct_risk_only(record["text"], int(risk))
        for record, risk in zip(records, raw)
    ], dtype=np.int64)


def _evidence_scores(records, prediction, calibration):
    result = np.empty(len(records), dtype=np.float32)
    for index, (record, risk) in enumerate(zip(records, prediction)):
        model_phrases = decode_model_evidence(
            record["text"], record["offsets"], record["start"], record["end"],
            threshold=float(calibration["threshold"]),
            max_tokens=int(calibration["max_tokens"]),
            end_policy=calibration["end_policy"], limit=5,
        )
        evidence = apply_evidence_policy(
            record["text"], int(risk), model_phrases,
            policy=calibration["cue_policy"], topk=int(calibration["topk"]),
        )
        result[index] = _post_phrase_f1(evidence, record["gold"])
    return result


def _grid(records, truth, probability, evidence_matrix, indices):
    rows = []
    for indicator_bias in (-0.30, -0.15, 0.0, 0.15):
        for behavior_bias in (-0.20, -0.10, 0.0, 0.10, 0.20):
            for attempt_bias in (0.0, 0.15, 0.30, 0.45, 0.60, 0.75):
                parameters = {
                    "indicator_bias": indicator_bias,
                    "behavior_bias": behavior_bias,
                    "attempt_bias": attempt_bias,
                }
                prediction = _predict(records, probability, parameters)
                phrase = evidence_matrix[np.arange(len(records)), prediction]
                subset = np.asarray(indices, dtype=int)
                risk = float(f1_score(
                    truth[subset], prediction[subset], average="weighted", zero_division=0
                ))
                phrase_f1 = float(phrase[subset].mean())
                rows.append({
                    **parameters, "risk_f1": risk, "phrase_f1": phrase_f1,
                    "task1": task1_score(risk, phrase_f1),
                })
    return sorted(rows, key=lambda row: row["task1"], reverse=True)


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
        raise ValueError("V10 and V4 strict folds differ")
    records = raw["records"]
    probability = _probabilities(dataset, train_idx, valid_idx, records)
    calibration = load_evidence_calibration()
    if calibration is None:
        raise FileNotFoundError("Adopted evidence calibration is required")
    truth = labels[valid_idx]
    local_groups = groups[valid_idx]
    indices = np.arange(len(records))
    baseline_parameters = {
        "indicator_bias": 0.0, "behavior_bias": 0.0, "attempt_bias": 0.0,
    }
    print("risk-v10: caching evidence scores for four possible risk labels...", flush=True)
    evidence_matrix = np.column_stack([
        _evidence_scores(
            records, np.full(len(records), risk_id, dtype=np.int64), calibration
        )
        for risk_id in range(config.NUM_RISK_CLASSES)
    ])
    baseline_prediction = _predict(records, probability, baseline_parameters)
    baseline_phrase = evidence_matrix[np.arange(len(records)), baseline_prediction]
    baseline_risk = float(f1_score(
        truth, baseline_prediction, average="weighted", zero_division=0
    ))
    baseline_task1 = task1_score(baseline_risk, float(baseline_phrase.mean()))

    print("risk-v10: nested user-level class-bias search...", flush=True)
    crossfit_prediction = np.empty(len(records), dtype=np.int64)
    crossfit_phrase = np.empty(len(records), dtype=np.float32)
    selections = []
    for fold, (fit, held) in enumerate(GroupKFold(n_splits=4).split(
        indices, groups=local_groups
    )):
        selected = _grid(records, truth, probability, evidence_matrix, fit)[0]
        parameters = {key: selected[key] for key in (
            "indicator_bias", "behavior_bias", "attempt_bias"
        )}
        prediction = _predict(records, probability, parameters)
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
            f"risk-v10: fold {fold + 1}/4 task1={selections[-1]['heldout_task1']:.4f}",
            flush=True,
        )
    production = {
        key: float(_median_member([row[key] for row in selections]))
        for key in ("indicator_bias", "behavior_bias", "attempt_bias")
    }
    fixed_prediction = _predict(records, probability, production)
    fixed_phrase = evidence_matrix[np.arange(len(records)), fixed_prediction]
    fixed_risk = float(f1_score(
        truth, fixed_prediction, average="weighted", zero_division=0
    ))
    crossfit_risk = float(f1_score(
        truth, crossfit_prediction, average="weighted", zero_division=0
    ))
    crossfit_task1 = task1_score(crossfit_risk, float(crossfit_phrase.mean()))
    fixed_task1 = task1_score(fixed_risk, float(fixed_phrase.mean()))

    rng = np.random.default_rng(config.SEED + 1010)
    unique = np.unique(local_groups); deltas = []
    for _ in range(2000):
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
    optimistic = _grid(records, truth, probability, evidence_matrix, indices)[0]
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
