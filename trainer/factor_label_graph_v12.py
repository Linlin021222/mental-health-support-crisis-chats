"""Leak-free label-graph decoding for the accepted Task 2 V3 ensemble.

The 24 factors are not independent.  This experiment learns a smoothed
pointwise-mutual-information graph from *outer-training users only*.  It also
selects a prevalence ratio separately for every label using the already OOF
predictions of those training users.  The selected parameters are then frozen
before scoring the untouched outer users.

Nothing in this module changes production inference automatically.
"""
from __future__ import annotations

import json

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

from configs.config import config
from inference.factor_nli import _rank_decode
from preprocess.preprocess import load_train_data
from trainer.factor_llm_lexical_v6 import _current_v3_probability


OUTPUT = config.OUTPUT_DIR / "factor_label_graph_v12"
RESULTS = OUTPUT / "cv_results.json"
TRAINING_VERSION = "factor-label-graph-crossfit-v12"

ALPHAS = np.asarray((0.0, 0.10, 0.20, 0.35, 0.50, 0.75), dtype=np.float32)
RATIOS = np.asarray((0.70, 0.85, 1.00, 1.10, 1.25, 1.40, 1.60), dtype=np.float32)


def _logit(probability):
    p = np.clip(np.asarray(probability, dtype=np.float64), 1e-5, 1.0 - 1e-5)
    return np.log(p) - np.log1p(-p)


def _sigmoid(value):
    value = np.clip(value, -20.0, 20.0)
    return 1.0 / (1.0 + np.exp(-value))


def _label_graph(targets):
    """Return a stable, directed PMI lift graph with no self edges."""
    y = np.asarray(targets, dtype=np.float64)
    # Beta smoothing is important for labels with fewer than ten positives.
    smoothing = 2.0
    n = float(len(y))
    marginal = (y.sum(0) + smoothing) / (n + 2.0 * smoothing)
    joint = (y.T @ y + smoothing) / (n + 2.0 * smoothing)
    lift = np.log(joint / np.maximum(marginal[:, None] * marginal[None, :], 1e-8))
    np.fill_diagonal(lift, 0.0)
    # Negative correlations in this very small dataset are much noisier than
    # positive co-occurrence, so propagate only well-supported positive lift.
    lift = np.clip(lift, 0.0, 1.5)
    column_norm = np.abs(lift).sum(0, keepdims=True)
    return lift / np.maximum(column_norm, 1.0), marginal


def _graph_probability(probability, graph, prevalence, alphas):
    centered = np.asarray(probability, dtype=np.float64) - prevalence[None, :]
    message = centered @ graph
    return _sigmoid(_logit(probability) + message * alphas[None, :]).astype(np.float32)


def _topk_one(scores, prevalence, ratio):
    count = max(1, int(round(len(scores) * float(prevalence) * float(ratio))))
    count = min(len(scores), count)
    chosen = np.argpartition(scores, len(scores) - count)[len(scores) - count:]
    prediction = np.zeros(len(scores), dtype=bool)
    prediction[chosen] = True
    return prediction


def _select_parameters(probability, targets, fit_idx):
    graph, prevalence = _label_graph(targets[fit_idx])
    parameters = np.zeros((config.NUM_FACTORS, 2), dtype=np.float32)
    for label in range(config.NUM_FACTORS):
        best = (-1.0, 0.0, 1.10)
        for alpha in ALPHAS:
            alpha_vector = np.zeros(config.NUM_FACTORS, dtype=np.float32)
            alpha_vector[label] = alpha
            score = _graph_probability(
                probability[fit_idx], graph, prevalence, alpha_vector
            )[:, label]
            for ratio in RATIOS:
                prediction = _topk_one(score, prevalence[label], ratio)
                value = f1_score(targets[fit_idx, label], prediction, zero_division=0)
                # Prefer less graph intervention, then a ratio nearer 1.1.
                key = (float(value), -float(alpha), -abs(float(ratio) - 1.10))
                old_key = (best[0], -best[1], -abs(best[2] - 1.10))
                if key > old_key:
                    best = (float(value), float(alpha), float(ratio))
        parameters[label] = (best[1], best[2])
    return graph, prevalence, parameters


def _predict(probability, graph, prevalence, parameters):
    adjusted = _graph_probability(probability, graph, prevalence, parameters[:, 0])
    return _rank_decode(adjusted, prevalence, parameters[:, 1]), adjusted


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame = load_train_data().reset_index(drop=True)
    targets = np.vstack(frame.factor_vector.to_numpy()).astype(np.int8)
    groups = frame.anon_user_id.astype(str).to_numpy()
    base, calibration = _current_v3_probability()
    folds = list(StratifiedGroupKFold(
        n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED,
    ).split(np.zeros(len(frame)), frame.risk_label.to_numpy(), groups))

    baseline_prediction = np.zeros_like(targets, dtype=bool)
    ratio_prediction = np.zeros_like(targets, dtype=bool)
    graph_prediction = np.zeros_like(targets, dtype=bool)
    rows, parameter_rows = [], []
    for fold, (fit_idx, valid_idx) in enumerate(folds):
        graph, prevalence, parameters = _select_parameters(base, targets, fit_idx)
        graph_pred, _ = _predict(base[valid_idx], graph, prevalence, parameters)
        ratio_parameters = parameters.copy(); ratio_parameters[:, 0] = 0.0
        ratio_pred, _ = _predict(base[valid_idx], graph, prevalence, ratio_parameters)
        baseline = _rank_decode(base[valid_idx], prevalence, 1.10)
        baseline_prediction[valid_idx] = baseline
        ratio_prediction[valid_idx] = ratio_pred
        graph_prediction[valid_idx] = graph_pred
        row = {
            "fold": fold,
            "baseline_macro_f1": float(f1_score(
                targets[valid_idx], baseline, average="macro", zero_division=0)),
            "per_label_ratio_macro_f1": float(f1_score(
                targets[valid_idx], ratio_pred, average="macro", zero_division=0)),
            "label_graph_macro_f1": float(f1_score(
                targets[valid_idx], graph_pred, average="macro", zero_division=0)),
        }
        rows.append(row); parameter_rows.append(parameters)
        print(json.dumps(row), flush=True)

    baseline_score = float(f1_score(
        targets, baseline_prediction, average="macro", zero_division=0))
    ratio_score = float(f1_score(
        targets, ratio_prediction, average="macro", zero_division=0))
    graph_score = float(f1_score(
        targets, graph_prediction, average="macro", zero_division=0))

    # Legitimate production parameters: selections use full-data OOF scores;
    # every score was generated by a model that did not train on that post/user.
    production_graph, production_prevalence, production_parameters = (
        _select_parameters(base, targets, np.arange(len(targets)))
    )
    np.savez_compressed(
        OUTPUT / "production_calibration.npz",
        graph=production_graph.astype(np.float32),
        prevalence=production_prevalence.astype(np.float32),
        alphas=production_parameters[:, 0], ratios=production_parameters[:, 1],
    )

    per_label = []
    for label, name in enumerate(config.FACTOR_LABELS):
        per_label.append({
            "label": name,
            "support": int(targets[:, label].sum()),
            "baseline_f1": float(f1_score(
                targets[:, label], baseline_prediction[:, label], zero_division=0)),
            "ratio_f1": float(f1_score(
                targets[:, label], ratio_prediction[:, label], zero_division=0)),
            "graph_f1": float(f1_score(
                targets[:, label], graph_prediction[:, label], zero_division=0)),
            "production_alpha": float(production_parameters[label, 0]),
            "production_ratio": float(production_parameters[label, 1]),
        })

    payload = {
        "training_version": TRAINING_VERSION,
        "evaluation": "five-fold nested user-disjoint calibration",
        "accepted_v3_baseline_macro_f1": baseline_score,
        "per_label_ratio_macro_f1": ratio_score,
        "label_graph_macro_f1": graph_score,
        "delta_ratio": ratio_score - baseline_score,
        "delta_graph": graph_score - baseline_score,
        "folds": rows,
        "per_label": per_label,
        "promising": bool(graph_score >= baseline_score + 0.005),
        "adopted": False,
        "previous_calibration_version": calibration.get("training_version"),
    }
    RESULTS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    return payload


if __name__ == "__main__":
    main()
