"""Nested user-level calibration of a MentalRoBERTa Task 1 risk expert."""
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
    EXTENDED_ATTEMPT_CUE, EXTENDED_BEHAVIOR_CUE, EXTENDED_IDEATION_CUE,
    apply_evidence_policy, correct_risk_only, decode_model_evidence,
    load_evidence_calibration,
)
from models.multitask_model_v2 import SuicideRiskMultiTaskModelV2
from preprocess.preprocess import load_train_data
from utils.task1_metric import task1_score


OUTPUT = config.OUTPUT_DIR / "task1_risk_v5"
CALIBRATION = OUTPUT / "calibration.json"


def _softmax(values, temperature):
    values = np.asarray(values, dtype=np.float32) / float(temperature)
    values -= values.max(1, keepdims=True)
    values = np.exp(values)
    return values / values.sum(1, keepdims=True)


def _cue_adjust(probability, text, boost):
    if boost <= 0:
        return probability
    logits = np.log(np.asarray(probability).clip(1e-7, 1.0))
    # Use the most severe explicit cue only. This preserves the task hierarchy
    # and avoids boosting both Ideation and Attempt for "tried to kill myself".
    if EXTENDED_ATTEMPT_CUE.search(text):
        logits[config.RISK_LABELS["Attempt"]] += boost
    elif EXTENDED_BEHAVIOR_CUE.search(text):
        logits[config.RISK_LABELS["Behavior"]] += boost
    elif EXTENDED_IDEATION_CUE.search(text):
        logits[config.RISK_LABELS["Ideation"]] += boost
    logits -= logits.max()
    result = np.exp(logits)
    return result / result.sum()


def _predictions(records, stable, mental_standard, mental_ordinal, parameters):
    mental = (
        (1.0 - parameters["mental_ordinal_weight"]) * mental_standard
        + parameters["mental_ordinal_weight"] * mental_ordinal
    )
    probability = (
        (1.0 - parameters["mental_weight"]) * stable
        + parameters["mental_weight"] * mental
    )
    prediction = []
    for index, record in enumerate(records):
        adjusted = _cue_adjust(probability[index], record["text"], parameters["cue_boost"])
        risk = correct_risk_only(record["text"], int(np.argmax(adjusted)))
        prediction.append(risk)
    return np.asarray(prediction, dtype=np.int64)


def _phrase_scores(records, predictions, model_phrases, evidence_calibration):
    scores = np.zeros(len(records), dtype=np.float32)
    for index, (record, risk) in enumerate(zip(records, predictions)):
        evidence = apply_evidence_policy(
            record["text"], int(risk), model_phrases[index],
            policy=evidence_calibration["cue_policy"],
            topk=int(evidence_calibration["topk"]),
        )
        scores[index] = _post_phrase_f1(evidence, record["gold"])
    return scores


def _evaluate(records, truth, stable, mental_standard, mental_ordinal,
              model_phrases, evidence_calibration, indices, parameters):
    prediction = _predictions(
        records, stable, mental_standard, mental_ordinal, parameters
    )
    phrase = _phrase_scores(records, prediction, model_phrases, evidence_calibration)
    indices = np.asarray(indices, dtype=int)
    risk_f1 = float(f1_score(
        truth[indices], prediction[indices], average="weighted", zero_division=0
    ))
    phrase_f1 = float(phrase[indices].mean())
    return {
        "risk_f1": risk_f1, "phrase_f1": phrase_f1,
        "task1": task1_score(risk_f1, phrase_f1),
        "prediction": prediction, "phrase_scores": phrase,
    }


def _grid(records, truth, stable, mental_standard, mental_ordinal,
          model_phrases, evidence_calibration, indices):
    rows = []
    for ordinal_weight in (0.0, 0.10, 0.20, 0.30, 0.40, 0.50):
        for mental_weight in (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40):
            for cue_boost in (0.0, 0.20, 0.40, 0.60):
                parameters = {
                    "mental_ordinal_weight": ordinal_weight,
                    "mental_weight": mental_weight,
                    "cue_boost": cue_boost,
                }
                metric = _evaluate(
                    records, truth, stable, mental_standard, mental_ordinal,
                    model_phrases, evidence_calibration, indices, parameters,
                )
                rows.append({
                    **parameters,
                    **{key: metric[key] for key in ("risk_f1", "phrase_f1", "task1")},
                })
    rows.sort(key=lambda row: row["task1"], reverse=True)
    return rows


def _median_member(values):
    median = float(np.median(values))
    return min(values, key=lambda value: (
        round(abs(float(value) - median), 12), float(value)
    ))


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    mental_file = config.OUTPUT_DIR / "task1_mental" / "strict_predictions.npz"
    raw_file = config.OUTPUT_DIR / "task1_evidence_v4" / "fold0_raw.pt"
    if not mental_file.exists():
        raise FileNotFoundError("Train the strict MentalRoBERTa risk expert first")
    dataset = SuicideRiskDataset(config.CACHE_DIR / "train_cache.pt")
    labels = np.asarray([int(row["risk_label"]) for row in dataset.data])
    groups = np.asarray([str(row["anon_user_id"]) for row in dataset.data])
    train_idx, valid_idx = next(StratifiedGroupKFold(
        n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED
    ).split(np.zeros(len(dataset)), labels, groups))
    raw = torch.load(raw_file, map_location="cpu", weights_only=False)
    if not np.array_equal(raw["valid_idx"], valid_idx):
        raise ValueError("Evidence and risk strict folds do not match")
    records = raw["records"]
    mental = np.load(mental_file)
    if not np.array_equal(mental["valid_indices"], valid_idx):
        raise ValueError("MentalRoBERTa and stable strict folds do not match")

    device = torch.device(config.DEVICE)
    v2_model = SuicideRiskMultiTaskModelV2().to(device)
    v2_model.load_state_dict(torch.load(
        config.OUTPUT_DIR / "task1_v2_strict_model.pt", map_location=device
    ))
    v2_records = collect(v2_model, dataset, valid_idx, device, v2=True)
    del v2_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    frame = load_train_data().reset_index(drop=True)
    vectorizer = _vectorizer()
    train_matrix = vectorizer.fit_transform(frame.text.iloc[train_idx])
    valid_matrix = vectorizer.transform(frame.text.iloc[valid_idx])
    lexical_model = LinearSVC(C=1.0, class_weight="balanced").fit(
        train_matrix, labels[train_idx]
    )
    lexical = _softmax(lexical_model.decision_function(valid_matrix), 0.5)
    old = np.vstack([record["old_probability"] for record in records])
    v2 = np.vstack([
        0.75 * record["standard"] + 0.25 * record["ordinal"]
        for record in v2_records
    ])
    transformer = 0.8 * old + 0.2 * v2
    stable = 0.7 * transformer + 0.3 * lexical

    evidence_calibration = load_evidence_calibration()
    if evidence_calibration is None:
        raise ValueError("Accepted evidence-v4 calibration is required")
    model_phrases = [
        decode_model_evidence(
            record["text"], record["offsets"], record["start"], record["end"],
            threshold=float(evidence_calibration["threshold"]),
            max_tokens=int(evidence_calibration["max_tokens"]),
            end_policy=evidence_calibration["end_policy"], limit=5,
        ) for record in records
    ]
    truth = labels[valid_idx]
    mental_standard = mental["standard"]
    mental_ordinal = mental["ordinal"]
    all_indices = np.arange(len(records))
    baseline_parameters = {
        "mental_ordinal_weight": 0.0, "mental_weight": 0.0, "cue_boost": 0.0,
    }
    baseline = _evaluate(
        records, truth, stable, mental_standard, mental_ordinal,
        model_phrases, evidence_calibration, all_indices, baseline_parameters,
    )

    local_groups = groups[valid_idx]
    crossfit_prediction = np.zeros(len(records), dtype=np.int64)
    crossfit_phrase = np.zeros(len(records), dtype=np.float32)
    selected = []
    for fold, (fit, held) in enumerate(GroupKFold(n_splits=4).split(
        all_indices, groups=local_groups
    )):
        parameter = _grid(
            records, truth, stable, mental_standard, mental_ordinal,
            model_phrases, evidence_calibration, fit,
        )[0]
        metric = _evaluate(
            records, truth, stable, mental_standard, mental_ordinal,
            model_phrases, evidence_calibration, held, parameter,
        )
        crossfit_prediction[held] = metric["prediction"][held]
        crossfit_phrase[held] = metric["phrase_scores"][held]
        selected.append({
            "fold": fold, **parameter,
            "heldout_risk_f1": metric["risk_f1"],
            "heldout_phrase_f1": metric["phrase_f1"],
            "heldout_task1": metric["task1"],
        })
    crossfit_risk = float(f1_score(
        truth, crossfit_prediction, average="weighted", zero_division=0
    ))
    crossfit_phrase_f1 = float(crossfit_phrase.mean())
    crossfit_task1 = task1_score(crossfit_risk, crossfit_phrase_f1)

    production = {
        "mental_ordinal_weight": float(_median_member([
            row["mental_ordinal_weight"] for row in selected
        ])),
        "mental_weight": float(_median_member([
            row["mental_weight"] for row in selected
        ])),
        "cue_boost": float(_median_member([row["cue_boost"] for row in selected])),
    }
    fixed = _evaluate(
        records, truth, stable, mental_standard, mental_ordinal,
        model_phrases, evidence_calibration, all_indices, production,
    )
    optimistic = _grid(
        records, truth, stable, mental_standard, mental_ordinal,
        model_phrases, evidence_calibration, all_indices,
    )[0]

    rng = np.random.default_rng(config.SEED + 1701)
    unique_users = np.unique(local_groups); deltas = []
    fixed_prediction = fixed["prediction"]; fixed_phrase = fixed["phrase_scores"]
    base_prediction = baseline["prediction"]; base_phrase = baseline["phrase_scores"]
    for _ in range(1000):
        sampled_users = rng.choice(unique_users, size=len(unique_users), replace=True)
        sampled = np.concatenate([
            np.flatnonzero(local_groups == user) for user in sampled_users
        ])
        base_risk = f1_score(
            truth[sampled], base_prediction[sampled], average="weighted", zero_division=0
        )
        new_risk = f1_score(
            truth[sampled], fixed_prediction[sampled], average="weighted", zero_division=0
        )
        deltas.append(task1_score(new_risk, fixed_phrase[sampled].mean())
                      - task1_score(base_risk, base_phrase[sampled].mean()))
    bootstrap = {
        "mean_delta": float(np.mean(deltas)),
        "p05_delta": float(np.quantile(deltas, 0.05)),
        "p95_delta": float(np.quantile(deltas, 0.95)),
        "positive_fraction": float(np.mean(np.asarray(deltas) > 0)),
    }
    adopted = bool(
        production["mental_weight"] > 0
        and crossfit_task1 >= baseline["task1"] + 0.004
        and fixed["task1"] >= baseline["task1"] + 0.004
        and bootstrap["positive_fraction"] >= 0.75
    )
    payload = {
        "training_version": "task1-risk-v5",
        "baseline": {key: baseline[key] for key in ("risk_f1", "phrase_f1", "task1")},
        "mental_standalone": json.loads(
            (config.OUTPUT_DIR / "task1_mental" / "strict_results.json").read_text(
                encoding="utf-8"
            )
        )["best"],
        "nested_crossfit": {
            "risk_f1": crossfit_risk, "phrase_f1": crossfit_phrase_f1,
            "task1": crossfit_task1, "folds": selected,
        },
        "fixed_production": {
            **production,
            **{key: fixed[key] for key in ("risk_f1", "phrase_f1", "task1")},
            "confusion": confusion_matrix(
                truth, fixed_prediction, labels=np.arange(config.NUM_RISK_CLASSES)
            ).tolist(),
        },
        "optimistic_full_holdout": optimistic,
        "user_cluster_bootstrap": bootstrap,
        "adopted": adopted,
    }
    (OUTPUT / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    CALIBRATION.write_text(json.dumps({
        "training_version": payload["training_version"], "adopted": adopted,
        **production,
        "strict_baseline_task1": baseline["task1"],
        "strict_crossfit_task1": crossfit_task1,
        "strict_fixed_task1": fixed["task1"],
        "bootstrap_positive_fraction": bootstrap["positive_fraction"],
    }, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2)); return payload


if __name__ == "__main__":
    main()
