"""Signed label-dependency and heterogeneous OOF stack for Task 2.

Unlike V12's unconditional positive co-occurrence propagation, this model
learns a separate sparse classifier for every target label.  It can assign
positive or negative weights to all 24 base label scores while also consulting
the same-label MentalRoBERTa, TF-IDF, NLI, retrieval and synthetic-distillation
experts.  Hyperparameters are selected in nested user-disjoint folds.
"""
from __future__ import annotations

import json

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize

from analyze_factor_knn_v5 import _retrieval_scores
from baseline import _vectorizer
from configs.config import config
from inference.factor_nli import _rank_decode
from preprocess.preprocess import load_train_data
from trainer.factor_llm_lexical_v6 import (
    _current_v3_probability, _fit_predict as _fit_lexical,
    _synthetic_rows as _v6_synthetic_rows,
)


OUTPUT = config.OUTPUT_DIR / "factor_signed_graph_stack_v21"
RESULTS = OUTPUT / "cv_results.json"
PREDICTIONS = OUTPUT / "nested_predictions.npz"
CALIBRATION = OUTPUT / "production_calibration.npz"
PRODUCTION_MODEL = OUTPUT / "production_models.joblib"
LEXICAL_MODEL = OUTPUT / "qwen_lexical_full.joblib"
RETRIEVAL_MODEL = OUTPUT / "retrieval_full.joblib"
TRAINING_VERSION = "signed-label-dependency-heterogeneous-stack-v21"
SEED = 212121

C_VALUES = (.003, .01, .03, .10, .30)
BLEND_WEIGHTS = (0., .10, .20, .35, .50)
RATIOS = (.90, 1.00, 1.10, 1.20)


def _load(path, key="probabilities"):
    values = np.load(path)[key].astype(np.float32)
    if values.shape[1] != config.NUM_FACTORS or not np.isfinite(values).all():
        raise ValueError(f"Invalid component {path}: {values.shape}")
    return values


def _components():
    current, _ = _current_v3_probability()
    factor_cv = np.load(config.OUTPUT_DIR / "factor_cv" / "factor_oof_predictions.npz")
    return {
        "current_v3": current,
        "mentalroberta": factor_cv["semantic"].astype(np.float32),
        "tfidf": factor_cv["cpu"].astype(np.float32),
        "old_nli": _load(config.OUTPUT_DIR / "factor_cross_encoder" / "oof_predictions.npz"),
        "prototype_nli": _load(config.OUTPUT_DIR / "factor_cross_encoder_v2" / "oof_predictions.npz"),
        "qwen_synthetic_lexical": _load(config.OUTPUT_DIR / "factor_llm_lexical_v6" / "oof_predictions.npz"),
        "knn": _load(config.OUTPUT_DIR / "factor_knn_v5" / "oof_predictions.npz", "knn_10"),
        "positive_retrieval": _load(config.OUTPUT_DIR / "factor_knn_v5" / "oof_predictions.npz", "positive_max"),
    }


def _logit(values):
    values = np.clip(np.asarray(values, dtype=np.float64), 1e-4, 1 - 1e-4)
    return np.log(values) - np.log1p(-values)


def _signed_graph(targets):
    """Shrinked signed phi correlations estimated from training labels only."""
    y = np.asarray(targets, dtype=np.float64)
    n = float(len(y)); counts = y.sum(0); prevalence = counts / max(n, 1.)
    joint = y.T @ y
    covariance = joint / max(n, 1.) - prevalence[:, None] * prevalence[None, :]
    variance = prevalence * (1. - prevalence)
    denominator = np.sqrt(np.maximum(variance[:, None] * variance[None, :], 1e-8))
    phi = covariance / denominator
    # Rare pair correlations are extremely volatile.  The empirical-Bayes
    # shrinkage keeps their sign but prevents one co-occurrence dominating.
    reliability = joint / (joint + 8.0)
    graph = np.clip(phi * reliability, -.75, .75)
    np.fill_diagonal(graph, 0.)
    return graph.astype(np.float32), prevalence.astype(np.float32)


def _feature_tensor(components, graph, prevalence):
    """rows x labels x features, with target-specific signed graph messages."""
    current = components["current_v3"]
    centered = current - prevalence[None, :]
    graph_message = centered @ graph
    all_base_logits = _logit(current).astype(np.float32)
    rows = []
    for label in range(config.NUM_FACTORS):
        same_label = np.column_stack([
            _logit(values[:, label]) for values in components.values()
        ]).astype(np.float32)
        # All V3 label activations let the sparse classifier learn adaptive,
        # directed dependencies.  The graph message adds a stable prior.
        features = np.column_stack((
            all_base_logits,
            same_label,
            graph_message[:, label],
            current[:, label] * graph_message[:, label],
        )).astype(np.float32)
        rows.append(features)
    return np.stack(rows, axis=1)


def _fit_label(x, y, c_value):
    if int(y.sum()) < 3 or int((1 - y).sum()) < 3:
        return None
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=float(c_value), l1_ratio=1.0, solver="liblinear",
            class_weight="balanced", max_iter=1000, random_state=SEED,
        ),
    )
    model.fit(x, y)
    return model


def _fit_predict(train_x, train_y, valid_x, c_value, fallback):
    result = np.empty((len(valid_x), config.NUM_FACTORS), dtype=np.float32)
    for label in range(config.NUM_FACTORS):
        model = _fit_label(train_x[:, label], train_y[:, label], c_value)
        result[:, label] = (fallback[:, label] if model is None else
                            model.predict_proba(valid_x[:, label])[:, 1])
    return result


def _train_production(frame, targets, components, graph, prevalence, parameters):
    """Fit deployable experts only after the nested adoption gate passes."""
    features = _feature_tensor(components, graph, prevalence)
    models = [
        _fit_label(features[:, label], targets[:, label], parameters["c"])
        for label in range(config.NUM_FACTORS)
    ]
    joblib.dump({
        "training_version": TRAINING_VERSION,
        "models": models, "graph": graph, "prevalence": prevalence,
        "weight": float(parameters["weight"]),
        "ratio": float(parameters["ratio"]),
        "components": list(components),
    }, PRODUCTION_MODEL)

    synthetic_texts, synthetic_labels = _v6_synthetic_rows()
    _fit_lexical(
        frame.text.astype(str).tolist(), targets, [],
        synthetic_texts, synthetic_labels, save_path=LEXICAL_MODEL,
    )
    vectorizer = _vectorizer()
    train_matrix = normalize(vectorizer.fit_transform(
        frame.text.astype(str).tolist()
    ))
    joblib.dump({
        "training_version": TRAINING_VERSION,
        "vectorizer": vectorizer, "matrix": train_matrix,
        "targets": targets,
    }, RETRIEVAL_MODEL)
    print(f"V21 production artifacts ready: {PRODUCTION_MODEL}", flush=True)


def _choose(components, targets, indices, risk, groups):
    """Inner OOF selection; outer validation users never choose parameters."""
    _, outer_prevalence = _signed_graph(targets[indices])
    splitter = StratifiedGroupKFold(
        n_splits=4, shuffle=True, random_state=config.SEED + 21,
    )
    meta_by_c = {c: np.zeros((len(indices), config.NUM_FACTORS), np.float32)
                 for c in C_VALUES}
    for fit_rel, valid_rel in splitter.split(
            np.zeros(len(indices)), risk[indices], groups[indices]):
        fit_idx, valid_idx = indices[fit_rel], indices[valid_rel]
        inner_graph, inner_prevalence = _signed_graph(targets[fit_idx])
        inner_features = _feature_tensor(
            components, inner_graph, inner_prevalence,
        )
        for c_value in C_VALUES:
            meta_by_c[c_value][valid_rel] = _fit_predict(
                inner_features[fit_idx], targets[fit_idx],
                inner_features[valid_idx], c_value,
                components["current_v3"][valid_idx],
            )
    base = components["current_v3"][indices]
    rows = []
    for c_value, meta in meta_by_c.items():
        for weight in BLEND_WEIGHTS:
            probability = (1. - weight) * base + weight * meta
            for ratio in RATIOS:
                prediction = _rank_decode(probability, outer_prevalence, ratio)
                rows.append({
                    "c": float(c_value), "weight": float(weight),
                    "ratio": float(ratio),
                    "score": float(f1_score(
                        targets[indices], prediction, average="macro", zero_division=0,
                    )),
                })
    # On exact ties prefer lower intervention and the established 1.10 ratio.
    best = max(rows, key=lambda row: (
        row["score"], -row["weight"], -abs(row["ratio"] - 1.10), -row["c"],
    ))
    return best, rows


def cross_validate():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame = load_train_data().reset_index(drop=True)
    targets = np.vstack(frame.factor_vector.to_numpy()).astype(np.int8)
    risk = frame.risk_label.to_numpy(np.int64)
    groups = frame.anon_user_id.astype(str).to_numpy()
    components = _components()
    expected = targets.shape
    for name, values in components.items():
        if values.shape != expected:
            raise ValueError(f"{name}: {values.shape}, expected {expected}")

    splitter = StratifiedGroupKFold(
        n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED,
    )
    baseline_prediction = np.zeros_like(targets, dtype=bool)
    candidate_prediction = np.zeros_like(targets, dtype=bool)
    candidate_probability = np.zeros_like(targets, dtype=np.float32)
    folds = []
    for fold, (fit_idx, valid_idx) in enumerate(
            splitter.split(np.zeros(len(frame)), risk, groups)):
        print(f"V21 outer fold {fold}: nested parameter selection", flush=True)
        best, grid = _choose(components, targets, fit_idx, risk, groups)
        graph, prevalence = _signed_graph(targets[fit_idx])
        features = _feature_tensor(components, graph, prevalence)
        meta = _fit_predict(
            features[fit_idx], targets[fit_idx], features[valid_idx], best["c"],
            components["current_v3"][valid_idx],
        )
        candidate = ((1. - best["weight"]) * components["current_v3"][valid_idx]
                     + best["weight"] * meta)
        baseline = _rank_decode(
            components["current_v3"][valid_idx], prevalence, 1.10,
        )
        prediction = _rank_decode(candidate, prevalence, best["ratio"])
        baseline_prediction[valid_idx] = baseline
        candidate_prediction[valid_idx] = prediction
        candidate_probability[valid_idx] = candidate
        row = {
            "fold": fold, **best,
            "baseline_macro_f1": float(f1_score(
                targets[valid_idx], baseline, average="macro", zero_division=0)),
            "candidate_macro_f1": float(f1_score(
                targets[valid_idx], prediction, average="macro", zero_division=0)),
            "inner_top5": sorted(grid, key=lambda x: x["score"], reverse=True)[:5],
        }
        folds.append(row); print(json.dumps(row, indent=2), flush=True)

    baseline = float(f1_score(
        targets, baseline_prediction, average="macro", zero_division=0,
    ))
    candidate = float(f1_score(
        targets, candidate_prediction, average="macro", zero_division=0,
    ))
    users = np.unique(groups); rng = np.random.default_rng(SEED); deltas = []
    for _ in range(3000):
        sampled = rng.choice(users, size=len(users), replace=True)
        positions = np.concatenate([np.flatnonzero(groups == user) for user in sampled])
        old = f1_score(targets[positions], baseline_prediction[positions],
                       average="macro", zero_division=0)
        new = f1_score(targets[positions], candidate_prediction[positions],
                       average="macro", zero_division=0)
        deltas.append(float(new - old))
    deltas = np.asarray(deltas)
    per_label = [{
        "label": config.ID2FACTOR[label], "support": int(targets[:, label].sum()),
        "baseline_f1": float(f1_score(
            targets[:, label], baseline_prediction[:, label], zero_division=0)),
        "candidate_f1": float(f1_score(
            targets[:, label], candidate_prediction[:, label], zero_division=0)),
    } for label in range(config.NUM_FACTORS)]
    bootstrap = {
        "mean_delta": float(deltas.mean()),
        "p05_delta": float(np.quantile(deltas, .05)),
        "p95_delta": float(np.quantile(deltas, .95)),
        "positive_fraction": float((deltas > 0).mean()),
    }
    adopted = bool(candidate >= baseline + .005
                   and bootstrap["positive_fraction"] >= .80)

    # Production parameters are useful only after the nested gate passes.
    production_best, _ = _choose(
        components, targets, np.arange(len(targets)), risk, groups,
    )
    production_graph, production_prevalence = _signed_graph(targets)
    np.savez_compressed(
        CALIBRATION, graph=production_graph, prevalence=production_prevalence,
        c=production_best["c"], weight=production_best["weight"],
        ratio=production_best["ratio"], adopted=adopted,
        training_version=TRAINING_VERSION,
    )
    if adopted:
        _train_production(
            frame, targets, components, production_graph,
            production_prevalence, production_best,
        )
    np.savez_compressed(
        PREDICTIONS, probabilities=candidate_probability,
        predictions=candidate_prediction, baseline_predictions=baseline_prediction,
        targets=targets,
    )
    payload = {
        "training_version": TRAINING_VERSION,
        "evaluation": "five-fold outer plus four-fold inner user-disjoint OOF",
        "components": list(components), "folds": folds,
        "baseline_nested_macro_f1": baseline,
        "candidate_nested_macro_f1": candidate,
        "delta": candidate - baseline,
        "per_label": per_label,
        "user_cluster_bootstrap": bootstrap,
        "production_parameters": production_best,
        "adopted": adopted,
    }
    RESULTS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    return payload


if __name__ == "__main__":
    cross_validate()
