"""Nested user-fold validation for shrinkage-based per-label expert routing."""
from __future__ import annotations

import ast
import json
from itertools import product
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
V3 = OUT / "factor_cross_encoder_v2"
ROUTING_DIR = OUT / "factor_routing_v4"
PRIOR_WEIGHTS = np.asarray([0.40, 0.15, 0.45], dtype=np.float64)
PRIOR_RATIO = 1.10
WEIGHT_GRID = [
    np.asarray([a, b, 1.0 - a - b], dtype=np.float64)
    for a in np.arange(0.0, 1.01, 0.10)
    for b in np.arange(0.0, 1.01 - a, 0.10)
]
RATIO_GRID = (0.80, 0.90, 1.00, 1.10, 1.25)
SHRINK_GRID = (10.0, 25.0, 50.0, 100.0, 200.0)
MARGIN_GRID = (0.0, 0.01, 0.02)


def _labels():
    tree = ast.parse((ROOT / "configs" / "config.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "FACTOR_LABELS"
            for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise ValueError("FACTOR_LABELS not found")


def _topk(scores, count):
    count = min(len(scores), max(1, int(count)))
    result = np.zeros(len(scores), dtype=bool)
    result[np.argpartition(scores, len(scores) - count)[len(scores) - count:]] = True
    return result


def _binary_f1(truth, prediction):
    truth = np.asarray(truth, dtype=bool); prediction = np.asarray(prediction, dtype=bool)
    tp = np.logical_and(truth, prediction).sum()
    fp = np.logical_and(~truth, prediction).sum()
    fn = np.logical_and(truth, ~prediction).sum()
    denominator = 2 * tp + fp + fn
    return 0.0 if denominator == 0 else float(2 * tp / denominator)


def _macro_f1(truth, prediction):
    return float(np.mean([_binary_f1(truth[:, j], prediction[:, j])
                          for j in range(truth.shape[1])]))


def _fit_raw_routes(components, truth, indices):
    """Fit independently best raw weights/ratio and record prior improvement."""
    routes = []
    n = len(indices)
    for label in range(truth.shape[1]):
        label_truth = truth[indices, label]
        prevalence = float(label_truth.mean())
        prior_scores = np.tensordot(PRIOR_WEIGHTS, components[:, indices, label], axes=(0, 0))
        prior_count = round(n * prevalence * PRIOR_RATIO)
        prior_f1 = _binary_f1(label_truth, _topk(prior_scores, prior_count))
        best = {"f1": prior_f1, "weights": PRIOR_WEIGHTS.copy(), "ratio": PRIOR_RATIO}
        for weights in WEIGHT_GRID:
            scores = np.tensordot(weights, components[:, indices, label], axes=(0, 0))
            for ratio in RATIO_GRID:
                score = _binary_f1(label_truth, _topk(scores, round(n * prevalence * ratio)))
                if score > best["f1"] + 1e-12:
                    best = {"f1": score, "weights": weights.copy(), "ratio": ratio}
        best["prior_f1"] = prior_f1
        best["support"] = int(label_truth.sum())
        routes.append(best)
    return routes


def _shrink_routes(raw_routes, shrink, minimum_gain):
    routes = []
    for raw in raw_routes:
        if raw["f1"] < raw["prior_f1"] + minimum_gain:
            coefficient = 0.0
        else:
            coefficient = raw["support"] / (raw["support"] + shrink)
        weights = coefficient * raw["weights"] + (1.0 - coefficient) * PRIOR_WEIGHTS
        ratio = coefficient * raw["ratio"] + (1.0 - coefficient) * PRIOR_RATIO
        routes.append({"weights": weights, "ratio": float(ratio), "coefficient": coefficient})
    return routes


def _apply_routes(components, truth, fit, valid, routes):
    prediction = np.zeros((len(valid), truth.shape[1]), dtype=bool)
    for label, route in enumerate(routes):
        prevalence = float(truth[fit, label].mean())
        scores = np.tensordot(
            route["weights"], components[:, valid, label], axes=(0, 0)
        )
        prediction[:, label] = _topk(
            scores, round(len(valid) * prevalence * route["ratio"])
        )
    return prediction


def main():
    labels = _labels()
    base_saved = np.load(OUT / "factor_cv" / "factor_oof_predictions.npz")
    truth = base_saved["targets"].astype(bool)
    base = 0.70 * base_saved["semantic"] + 0.30 * base_saved["cpu"]
    old = np.load(OUT / "factor_cross_encoder" / "oof_predictions.npz")["probabilities"]
    new = np.load(V3 / "oof_predictions.npz")["probabilities"]
    components = np.stack([base, old, new])
    folds = [np.load(V3 / f"fold{fold}_valid.npz")["valid_indices"].astype(int)
             for fold in range(5)]
    all_indices = np.arange(len(truth))

    nested_prediction = np.zeros_like(truth)
    selections = []
    for outer_fold, outer_valid in enumerate(folds):
        outer_fit = np.setdiff1d(all_indices, outer_valid)
        candidate_scores = []
        for shrink, margin in product(SHRINK_GRID, MARGIN_GRID):
            inner_truth, inner_prediction = [], []
            for inner_fold, inner_valid in enumerate(folds):
                if inner_fold == outer_fold:
                    continue
                inner_fit = np.setdiff1d(outer_fit, inner_valid)
                raw = _fit_raw_routes(components, truth, inner_fit)
                routes = _shrink_routes(raw, shrink, margin)
                inner_truth.append(truth[inner_valid])
                inner_prediction.append(
                    _apply_routes(components, truth, inner_fit, inner_valid, routes)
                )
            candidate_scores.append({
                "shrink": shrink, "minimum_gain": margin,
                "macro_f1": _macro_f1(np.vstack(inner_truth), np.vstack(inner_prediction)),
            })
        selected = max(candidate_scores, key=lambda item: item["macro_f1"])
        raw = _fit_raw_routes(components, truth, outer_fit)
        routes = _shrink_routes(raw, selected["shrink"], selected["minimum_gain"])
        nested_prediction[outer_valid] = _apply_routes(
            components, truth, outer_fit, outer_valid, routes
        )
        selections.append({"fold": outer_fold, **selected})

    nested_score = _macro_f1(truth, nested_prediction)
    # Production parameters are fitted once on all OOF-labelled posts, using
    # the median strictly nested hyperparameters rather than the best full-OOF
    # setting.
    production_shrink = float(np.median([item["shrink"] for item in selections]))
    production_margin = float(np.median([item["minimum_gain"] for item in selections]))
    raw = _fit_raw_routes(components, truth, all_indices)
    production_routes = _shrink_routes(raw, production_shrink, production_margin)
    production_prediction = _apply_routes(
        components, truth, all_indices, all_indices, production_routes
    )
    production_score = _macro_f1(truth, production_prediction)
    calibration = {
        "training_version": "per-label-routing-v4",
        "adopted": nested_score >= 0.594,
        "nested_crossfit_macro_f1": nested_score,
        "v3_global_crossfit_reference": 0.5911697006932748,
        "production_oof_macro_f1": production_score,
        "shrink": production_shrink,
        "minimum_gain": production_margin,
        "prior_weights": PRIOR_WEIGHTS.tolist(),
        "prior_ratio": PRIOR_RATIO,
        "weights": [route["weights"].tolist() for route in production_routes],
        "prevalence_ratios": [route["ratio"] for route in production_routes],
        "training_prevalence": truth.mean(0).tolist(),
        "labels": labels,
    }
    per_label = []
    global_path = V3 / "error_analysis.json"
    global_analysis = json.loads(global_path.read_text(encoding="utf-8"))
    for label, before in zip(labels, global_analysis["per_label"]):
        index = labels.index(label)
        after = _binary_f1(truth[:, index], nested_prediction[:, index])
        per_label.append({
            "label": label, "support": int(truth[:, index].sum()),
            "global_v3_crossfit_f1": before["v3_crossfit_f1"],
            "routing_nested_crossfit_f1": after,
            "delta": after - before["v3_crossfit_f1"],
        })
    result = {
        "calibration": calibration, "outer_selections": selections,
        "per_label": per_label,
    }
    ROUTING_DIR.mkdir(parents=True, exist_ok=True)
    (ROUTING_DIR / "analysis.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    (ROUTING_DIR / "calibration.json").write_text(
        json.dumps(calibration, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
