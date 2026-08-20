"""Leak-aware Task 2 decoding/stacking experiments over saved five-fold OOF scores.

This script is deliberately CPU-only: it evaluates whether inexpensive decoding
or label-dependency ideas deserve a new GPU training run.  Every reported
``crossfit`` prediction is produced with parameters fitted on the other four
user-disjoint folds.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "outputs"
N_FOLDS = 5
FACTOR_LABELS = list(json.loads(
    (OUTPUT_ROOT / "factor_cross_encoder" / "per_label_adopted.json").read_text(encoding="utf-8")
)["per_label"].keys())
NUM_FACTORS = len(FACTOR_LABELS)
OUT_DIR = OUTPUT_ROOT / "factor_stacking_v3"
RESULT_FILE = OUT_DIR / "analysis.json"


def _fold_ids(n_rows: int) -> np.ndarray:
    result = np.full(n_rows, -1, dtype=np.int8)
    for fold in range(N_FOLDS):
        saved = np.load(OUTPUT_ROOT / "factor_cross_encoder" / f"fold{fold}_valid.npz")
        result[saved["valid_indices"].astype(int)] = fold
    if np.any(result < 0):
        raise ValueError("Some OOF rows were not assigned to a fold")
    return result


def _rank_decode(probabilities, prevalence, ratio=1.10):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    result = np.zeros_like(probabilities, dtype=bool)
    n = len(probabilities)
    for label in range(probabilities.shape[1]):
        count = max(1, min(n, int(round(n * float(prevalence[label]) * ratio))))
        chosen = np.argpartition(probabilities[:, label], n - count)[n - count:]
        result[chosen, label] = True
    return result


def _score(targets, predictions):
    return float(_per_label_f1(targets, predictions).mean())


def _per_label_f1(targets, predictions):
    targets = np.asarray(targets, dtype=bool)
    predictions = np.asarray(predictions, dtype=bool)
    tp = np.logical_and(targets, predictions).sum(0).astype(float)
    fp = np.logical_and(~targets, predictions).sum(0).astype(float)
    fn = np.logical_and(targets, ~predictions).sum(0).astype(float)
    denominator = 2.0 * tp + fp + fn
    return np.divide(2.0 * tp, denominator, out=np.zeros_like(tp), where=denominator > 0)


def _binary_f1(targets, predictions):
    return float(_per_label_f1(np.asarray(targets)[:, None], np.asarray(predictions)[:, None])[0])


def _clip_logit(probability):
    probability = np.clip(probability, 1e-5, 1 - 1e-5)
    return np.log(probability / (1.0 - probability))


def _local_features(base, cross, nli):
    lb, lc, ln = _clip_logit(base), _clip_logit(cross), _clip_logit(nli)
    return np.stack([
        lb, lc, ln, 0.5 * (lb + lc), lc - lb,
        base * cross, np.maximum(base, cross), np.minimum(base, cross),
    ], axis=-1)


def _crossfit_fixed(base, cross, targets, fold_ids, weight=0.5, ratio=1.10):
    result = np.zeros_like(targets, dtype=bool)
    for fold in range(N_FOLDS):
        fit, valid = fold_ids != fold, fold_ids == fold
        probability = (1.0 - weight) * base[valid] + weight * cross[valid]
        result[valid] = _rank_decode(probability, targets[fit].mean(0), ratio)
    return result


def _per_label_weight_crossfit(base, cross, targets, fold_ids, shrink_support=0):
    result = np.zeros_like(targets, dtype=bool)
    parameters = []
    weight_grid = np.linspace(0.0, 1.0, 11)
    ratio_grid = (0.90, 1.00, 1.10, 1.25)
    for fold in range(N_FOLDS):
        fit, valid = fold_ids != fold, fold_ids == fold
        prevalence = targets[fit].mean(0)
        weights, ratios = np.zeros(NUM_FACTORS), np.zeros(NUM_FACTORS)
        for label in range(NUM_FACTORS):
            candidates = []
            for weight in weight_grid:
                probability = (1.0 - weight) * base[fit, label] + weight * cross[fit, label]
                for ratio in ratio_grid:
                    count = max(1, min(int(fit.sum()), int(round(fit.sum() * prevalence[label] * ratio))))
                    chosen = np.argpartition(probability, len(probability) - count)[len(probability) - count:]
                    prediction = np.zeros(len(probability), dtype=bool); prediction[chosen] = True
                    candidates.append((_binary_f1(targets[fit, label], prediction), weight, ratio))
            _, weight, ratio = max(candidates, key=lambda item: (item[0], -abs(item[1] - 0.5), -abs(item[2] - 1.1)))
            support = int(targets[fit, label].sum())
            if shrink_support:
                reliability = support / (support + shrink_support)
                weight = reliability * weight + (1.0 - reliability) * 0.5
                ratio = reliability * ratio + (1.0 - reliability) * 1.1
            weights[label], ratios[label] = weight, ratio
            probability = (1.0 - weight) * base[valid, label] + weight * cross[valid, label]
            count = max(1, min(int(valid.sum()), int(round(valid.sum() * prevalence[label] * ratio))))
            chosen = np.argpartition(probability, len(probability) - count)[len(probability) - count:]
            valid_rows = np.flatnonzero(valid)
            result[valid_rows[chosen], label] = True
        parameters.append({"fold": fold, "weights": weights.tolist(), "ratios": ratios.tolist()})
    return result, parameters


def _ridge_ranker(x_fit, y_fit, x_valid, penalty):
    """Balanced linear ranker; only ordering matters for prior-rank decoding."""
    mean = x_fit.mean(0)
    scale = x_fit.std(0).clip(min=1e-4)
    train = (x_fit - mean) / scale
    valid = (x_valid - mean) / scale
    train = np.column_stack([np.ones(len(train)), train])
    valid = np.column_stack([np.ones(len(valid)), valid])
    positives = max(1, int(y_fit.sum()))
    negatives = max(1, len(y_fit) - positives)
    weights = np.where(y_fit > 0, len(y_fit) / (2.0 * positives), len(y_fit) / (2.0 * negatives))
    lhs = train.T @ (train * weights[:, None])
    regulariser = np.eye(lhs.shape[0]) * penalty
    regulariser[0, 0] = 1e-6
    coefficients = np.linalg.solve(lhs + regulariser, train.T @ (weights * y_fit))
    return valid @ coefficients


def _linear_stacker_crossfit(base, cross, nli, targets, fold_ids, graph=False, penalty=20.0):
    local = _local_features(base, cross, nli)
    result = np.zeros_like(targets, dtype=bool)
    probabilities = np.zeros_like(base, dtype=np.float64)
    for fold in range(N_FOLDS):
        fit, valid = fold_ids != fold, fold_ids == fold
        for label in range(NUM_FACTORS):
            x_fit = local[fit, label]
            x_valid = local[valid, label]
            if graph:
                # Strong regularisation is essential: the added logits represent
                # co-occurring factors and are not direct evidence for this label.
                graph_fit = np.concatenate([_clip_logit(base[fit]), _clip_logit(cross[fit])], axis=1)
                graph_valid = np.concatenate([_clip_logit(base[valid]), _clip_logit(cross[valid])], axis=1)
                x_fit = np.concatenate([x_fit, graph_fit], axis=1)
                x_valid = np.concatenate([x_valid, graph_valid], axis=1)
            probabilities[valid, label] = _ridge_ranker(
                x_fit, targets[fit, label].astype(float), x_valid, penalty
            )
        result[valid] = _rank_decode(probabilities[valid], targets[fit].mean(0), 1.10)
    return result, probabilities


def _graph_residual_crossfit(base, cross, targets, fold_ids, alpha):
    mixed = 0.5 * (base + cross)
    result = np.zeros_like(targets, dtype=bool)
    adjusted = np.zeros_like(mixed, dtype=np.float64)
    for fold in range(N_FOLDS):
        fit, valid = fold_ids != fold, fold_ids == fold
        y = targets[fit].astype(np.float64)
        prior = (y.sum(0) + 1.0) / (len(y) + 2.0)
        cooccurrence = y.T @ y
        conditional = (cooccurrence + 1.0) / (y.sum(0)[:, None] + 2.0)
        lift = np.log(np.clip(conditional / prior[None, :], 0.25, 4.0))
        np.fill_diagonal(lift, 0.0)
        graph_signal = ((mixed[valid] - prior[None, :]) @ lift) / np.sqrt(NUM_FACTORS)
        adjusted[valid] = 1.0 / (1.0 + np.exp(-(_clip_logit(mixed[valid]) + alpha * graph_signal)))
        result[valid] = _rank_decode(adjusted[valid], prior, 1.10)
    return result, adjusted


def _cardinality_features(probabilities, base, cross):
    ordered = np.sort(probabilities, axis=1)[:, ::-1]
    return np.column_stack([
        ordered[:, :12], probabilities.sum(1), probabilities.mean(1), probabilities.std(1),
        ordered[:, 0] - ordered[:, 1], ordered[:, 2] - ordered[:, 3],
        (probabilities >= 0.50).sum(1), (probabilities >= 0.35).sum(1),
        np.abs(base - cross).mean(1), (base * cross).sum(1),
    ])


def _ridge_regression(x_fit, y_fit, x_valid, penalty=50.0):
    mean = x_fit.mean(0); scale = x_fit.std(0).clip(min=1e-4)
    train = np.column_stack([np.ones(len(x_fit)), (x_fit - mean) / scale])
    valid = np.column_stack([np.ones(len(x_valid)), (x_valid - mean) / scale])
    regulariser = np.eye(train.shape[1]) * penalty
    regulariser[0, 0] = 1e-6
    coefficients = np.linalg.solve(train.T @ train + regulariser, train.T @ y_fit)
    return valid @ coefficients


def _adjust_to_cardinality(initial, probabilities, base, cross, targets, fold_ids, max_changes):
    result = initial.copy()
    predicted_counts = np.zeros(len(targets), dtype=int)
    count_truth = targets.sum(1)
    features = _cardinality_features(probabilities, base, cross)
    maes = []
    for fold in range(N_FOLDS):
        fit, valid = fold_ids != fold, fold_ids == fold
        raw = _ridge_regression(features[fit], count_truth[fit], features[valid])
        counts = np.clip(np.rint(raw), 0, 12).astype(int)
        predicted_counts[valid] = counts
        maes.append(float(np.abs(count_truth[valid] - raw).mean()))

        # Calibrate score margins on the fit portion. This makes scores from
        # frequent and rare labels comparable when adding/removing labels per row.
        thresholds = np.quantile(
            probabilities[fit], 1.0 - np.clip(targets[fit].mean(0) * 1.10, 1 / fit.sum(), 1.0),
            axis=0,
        ).diagonal() if False else np.array([
            np.quantile(probabilities[fit, j], 1.0 - min(1.0, targets[fit, j].mean() * 1.10))
            for j in range(NUM_FACTORS)
        ])
        spread = np.std(probabilities[fit], axis=0).clip(min=0.03)
        margins = (probabilities[valid] - thresholds[None, :]) / spread[None, :]
        valid_rows = np.flatnonzero(valid)
        for local_row, global_row in enumerate(valid_rows):
            current = int(result[global_row].sum())
            desired = int(counts[local_row])
            delta = int(np.clip(desired - current, -max_changes, max_changes))
            if delta > 0:
                available = np.flatnonzero(~result[global_row])
                add = available[np.argsort(margins[local_row, available])[-delta:]]
                result[global_row, add] = True
            elif delta < 0:
                available = np.flatnonzero(result[global_row])
                remove = available[np.argsort(margins[local_row, available])[: -delta]]
                result[global_row, remove] = False
    return result, predicted_counts, float(np.mean(maes))


def main():
    base_saved = np.load(OUTPUT_ROOT / "factor_cv" / "factor_oof_predictions.npz")
    cross_saved = np.load(OUTPUT_ROOT / "factor_cross_encoder" / "oof_predictions.npz")
    nli = np.load(OUTPUT_ROOT / "factor_nli" / "train_probabilities.npz")["probabilities"]
    targets = cross_saved["targets"].astype(np.int8)
    base = 0.70 * base_saved["semantic"] + 0.30 * base_saved["cpu"]
    cross = cross_saved["probabilities"]
    fold_ids = _fold_ids(len(targets))
    prevalence = targets.mean(0)
    mixed = 0.5 * (base + cross)

    predictions = {}
    predictions["adopted_full_oof"] = _rank_decode(mixed, prevalence, 1.10)
    predictions["adopted_fixed_crossfit"] = _crossfit_fixed(base, cross, targets, fold_ids)

    parameter_payload = {}
    for shrink in (0, 10, 25, 50):
        name = f"per_label_weight_crossfit_shrink{shrink}"
        predictions[name], parameter_payload[name] = _per_label_weight_crossfit(
            base, cross, targets, fold_ids, shrink_support=shrink
        )

    for penalty in (5.0, 20.0, 50.0, 100.0):
        name = f"local_linear_crossfit_l2_{penalty}"
        predictions[name], _ = _linear_stacker_crossfit(
            base, cross, nli, targets, fold_ids, graph=False, penalty=penalty
        )
    for penalty in (20.0, 50.0, 100.0, 250.0):
        name = f"graph_linear_crossfit_l2_{penalty}"
        predictions[name], _ = _linear_stacker_crossfit(
            base, cross, nli, targets, fold_ids, graph=True, penalty=penalty
        )
    for alpha in (0.05, 0.10, 0.20, 0.35):
        name = f"cooccurrence_residual_crossfit_a{alpha}"
        predictions[name], _ = _graph_residual_crossfit(base, cross, targets, fold_ids, alpha)

    fixed = predictions["adopted_fixed_crossfit"]
    for changes in (1, 2, 4, 24):
        name = f"cardinality_adjust_crossfit_max{changes}"
        prediction, counts, mae = _adjust_to_cardinality(
            fixed, mixed, base, cross, targets, fold_ids, changes
        )
        predictions[name] = prediction
        parameter_payload[name] = {
            "count_mae": mae,
            "mean_predicted_count": float(counts.mean()),
            "mean_true_count": float(targets.sum(1).mean()),
        }

    scores = {name: _score(targets, prediction) for name, prediction in predictions.items()}
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_name = ordered[0][0]
    per_label = _per_label_f1(targets, predictions[best_name])
    adopted_per_label = _per_label_f1(targets, predictions["adopted_full_oof"])
    result = {
        "scores": dict(ordered),
        "best": best_name,
        "parameters": parameter_payload,
        "best_per_label": {
            FACTOR_LABELS[j]: {
                "support": int(targets[:, j].sum()),
                "adopted_full_oof_f1": float(adopted_per_label[j]),
                "candidate_crossfit_f1": float(per_label[j]),
                "delta": float(per_label[j] - adopted_per_label[j]),
            }
            for j in range(NUM_FACTORS)
        },
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_FILE.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"scores": dict(ordered), "best": best_name}, indent=2))


if __name__ == "__main__":
    main()
