"""Nested rank-space fusion for the definition features produced by V23b."""
from __future__ import annotations

import json

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from configs.config import config
from inference.factor_nli import _rank_decode
from preprocess.preprocess import load_train_data
from trainer.factor_definition_ranker_v23 import (
    FEATURES, TRAINING_VERSION as FEATURE_VERSION, _fit_rankers,
)
from trainer.factor_llm_lexical_v6 import _current_v3_probability


OUTPUT = config.OUTPUT_DIR / "factor_definition_rank_fusion_v24"
RESULTS = OUTPUT / "fold0_results.json"
TRAINING_VERSION = "nested-definition-rank-fusion-v24"
WEIGHTS = (0.0, 0.10, 0.20, 0.30, 0.40)


def _percentile_rank(values):
    values = np.asarray(values)
    result = np.empty_like(values, dtype=np.float32)
    denominator = max(1, len(values) - 1)
    for label in range(values.shape[1]):
        order = np.argsort(values[:, label], kind="mergesort")
        result[order, label] = np.arange(len(values), dtype=np.float32) / denominator
    return result


def _label_f1(truth, score, count):
    count = max(1, min(len(score), int(count)))
    prediction = np.zeros(len(score), dtype=bool)
    prediction[np.argpartition(score, len(score) - count)[len(score) - count:]] = True
    return float(f1_score(truth, prediction, zero_division=0))


def _inner_oof(features, targets, risk, groups, outer_train):
    probability = np.zeros((len(outer_train), config.NUM_FACTORS), dtype=np.float32)
    local_risk = risk[outer_train]
    local_groups = groups[outer_train]
    splitter = StratifiedGroupKFold(
        n_splits=4, shuffle=True, random_state=config.SEED + 2400,
    )
    for inner_fit, inner_valid in splitter.split(
        np.zeros(len(outer_train)), local_risk, local_groups
    ):
        fit = outer_train[inner_fit]
        valid = outer_train[inner_valid]
        fold_probability, _ = _fit_rankers(features, targets, fit, valid)
        probability[inner_valid] = fold_probability
    return probability


def _select_weights(current, definition, targets, prevalence):
    current_rank = _percentile_rank(current)
    definition_rank = _percentile_rank(definition)
    selected, audit = [], []
    n = len(targets)
    for label in range(config.NUM_FACTORS):
        support = int(targets[:, label].sum())
        count = max(1, int(round(n * float(prevalence[label]) * 1.10)))
        baseline = _label_f1(targets[:, label], current_rank[:, label], count)
        current_auc = (float(roc_auc_score(targets[:, label], current_rank[:, label]))
                       if 0 < support < n else 0.5)
        definition_auc = (float(roc_auc_score(targets[:, label], definition_rank[:, label]))
                          if 0 < support < n else 0.5)
        rows = []
        for weight in WEIGHTS:
            score = ((1.0 - weight) * current_rank[:, label]
                     + weight * definition_rank[:, label])
            rows.append((
                _label_f1(targets[:, label], score, count), weight,
            ))
        best_f1, best_weight = max(rows, key=lambda row: (row[0], -row[1]))
        # Rare labels need a larger observed gain because one example can move
        # their F1 dramatically. AUC is used only as a stability gate, never
        # to choose a weight on the untouched outer fold.
        required = 0.08 if support < 12 else 0.02
        if (best_f1 < baseline + required
                or definition_auc < current_auc + 0.002):
            best_weight = 0.0
            best_f1 = baseline
        selected.append(float(best_weight))
        audit.append({
            "label": config.ID2FACTOR[label], "support": support,
            "baseline_f1": baseline, "selected_f1": best_f1,
            "current_auc": current_auc, "definition_auc": definition_auc,
            "selected_weight": float(best_weight),
        })
    return np.asarray(selected, dtype=np.float32), audit


def train_fold0():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if not FEATURES.exists():
        raise FileNotFoundError(
            f"Missing {FEATURES}; run --mode factor-definition-ranker-v23-fold0 first."
        )
    saved = np.load(FEATURES)
    if str(saved["training_version"]) != FEATURE_VERSION:
        raise RuntimeError("V23b feature cache version mismatch")
    features = saved["features"].astype(np.float32)
    frame = load_train_data().reset_index(drop=True)
    targets = np.vstack(frame.factor_vector.to_numpy()).astype(np.int8)
    risk = frame.risk_label.to_numpy(dtype=np.int64)
    groups = frame.anon_user_id.astype(str).to_numpy()
    outer_train, outer_valid = next(StratifiedGroupKFold(
        n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED,
    ).split(np.zeros(len(frame)), risk, groups))

    current, _ = _current_v3_probability()
    inner_definition = _inner_oof(
        features, targets, risk, groups, outer_train
    )
    prevalence = targets[outer_train].mean(0)
    weights, selection = _select_weights(
        current[outer_train], inner_definition, targets[outer_train], prevalence
    )
    outer_definition, _ = _fit_rankers(
        features, targets, outer_train, outer_valid
    )
    current_rank = _percentile_rank(current[outer_valid])
    definition_rank = _percentile_rank(outer_definition)
    candidate_score = (
        (1.0 - weights[None, :]) * current_rank
        + weights[None, :] * definition_rank
    )
    baseline_prediction = _rank_decode(current[outer_valid], prevalence, 1.10)
    candidate_prediction = _rank_decode(candidate_score, prevalence, 1.10)
    baseline = float(f1_score(
        targets[outer_valid], baseline_prediction, average="macro", zero_division=0
    ))
    candidate = float(f1_score(
        targets[outer_valid], candidate_prediction, average="macro", zero_division=0
    ))

    global_diagnostic = []
    for weight in WEIGHTS:
        score = (1.0 - weight) * current_rank + weight * definition_rank
        prediction = _rank_decode(score, prevalence, 1.10)
        global_diagnostic.append({
            "weight": weight,
            "macro_f1": float(f1_score(
                targets[outer_valid], prediction, average="macro", zero_division=0
            )),
        })
    per_label = []
    for label in range(config.NUM_FACTORS):
        per_label.append({
            "label": config.ID2FACTOR[label],
            "support": int(targets[outer_valid, label].sum()),
            "selected_weight_from_inner_oof": float(weights[label]),
            "baseline_f1": float(f1_score(
                targets[outer_valid, label], baseline_prediction[:, label], zero_division=0
            )),
            "candidate_f1": float(f1_score(
                targets[outer_valid, label], candidate_prediction[:, label], zero_division=0
            )),
        })
    payload = {
        "training_version": TRAINING_VERSION,
        "evaluation": "outer user fold untouched; per-label weights selected on four-fold inner group OOF",
        "decoder": {"prevalence_ratio": 1.10, "score_space": "per-label percentile rank"},
        "selected_nonzero_labels": int((weights > 0).sum()),
        "selected_weights": weights.astype(float).tolist(),
        "baseline_macro_f1": baseline,
        "candidate_macro_f1": candidate,
        "delta": candidate - baseline,
        "promising_for_full_oof": bool(candidate >= baseline + .005),
        "adopted": False,
        "inner_selection": selection,
        "outer_per_label": per_label,
        "outer_global_weight_diagnostic_not_for_selection": global_diagnostic,
    }
    RESULTS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    return payload


if __name__ == "__main__":
    train_fold0()
