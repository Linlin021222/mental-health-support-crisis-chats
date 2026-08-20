"""Leak-aware heterogeneous OOF stacking for Subtask 2 (V8).

Every input feature is itself an out-of-fold prediction.  The second-level
models are fitted on four user-disjoint folds and evaluated on the fifth, so
the reported score does not train on the post or user it evaluates.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

from configs.config import config
from inference.factor_nli import _rank_decode
from preprocess.preprocess import load_train_data
from trainer.factor_llm_lexical_v6 import _current_v3_probability


OUTPUT = config.OUTPUT_DIR / "factor_heterogeneous_stack_v8"
RESULTS_FILE = OUTPUT / "cv_results.json"
PREDICTION_FILE = OUTPUT / "nested_predictions.npz"
TRAINING_VERSION = "heterogeneous-percentile-stack-v8"
SEED = 808080
CS = (0.10, 0.30, 1.00, 3.00, 10.00)
RATIOS = (0.90, 1.00, 1.10, 1.20)


def _load_npz(path, key="probabilities"):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return np.load(path)[key].astype(np.float32)


def _components():
    current, _ = _current_v3_probability()
    frame = load_train_data().reset_index(drop=True)
    targets = np.vstack(frame.factor_vector.to_numpy()).astype(np.int8)
    components = {
        "current_v3": current,
        "prototype_cross": _load_npz(
            config.OUTPUT_DIR / "factor_cross_encoder_v2" / "oof_predictions.npz"
        ),
        "old_cross": _load_npz(
            config.OUTPUT_DIR / "factor_cross_encoder" / "oof_predictions.npz"
        ),
        "mhlat": _load_npz(
            config.OUTPUT_DIR / "factor_mhlat_v4" / "oof_predictions.npz"
        ),
        "lexical": _load_npz(
            config.OUTPUT_DIR / "factor_llm_lexical_v6" / "oof_predictions.npz"
        ),
        "nli": _load_npz(
            config.OUTPUT_DIR / "factor_nli" / "train_probabilities.npz"
        ),
        "knn10": _load_npz(
            config.OUTPUT_DIR / "factor_knn_v5" / "oof_predictions.npz", "knn_10"
        ),
        "positive_retrieval": _load_npz(
            config.OUTPUT_DIR / "factor_knn_v5" / "oof_predictions.npz", "positive_max"
        ),
    }
    expected = targets.shape
    for name, values in components.items():
        if values.shape != expected:
            raise ValueError(f"{name} shape {values.shape}, expected {expected}")
        if not np.isfinite(values).all():
            raise ValueError(f"{name} contains non-finite values")
    return components, targets.astype(np.int8)


def _percentile(values):
    """Convert each model/label score to a distribution-free percentile."""
    values = np.asarray(values)
    result = np.empty_like(values, dtype=np.float32)
    n = len(values)
    if n <= 1:
        result.fill(0.5)
        return result
    for feature in range(values.shape[1]):
        order = np.argsort(values[:, feature], kind="mergesort")
        result[order, feature] = np.arange(n, dtype=np.float32) / (n - 1)
    return result


def _features(components, indices):
    # Shape: rows x labels x heterogeneous models. Ranking is performed only
    # inside the requested partition, preventing held-out distribution values
    # from influencing train features.
    return np.stack([_percentile(values[indices]) for values in components.values()], axis=-1)


def _fit_predict(train_x, train_y, valid_x, c_value):
    probability = np.empty_like(valid_y := np.zeros(
        (len(valid_x), config.NUM_FACTORS), dtype=np.float32
    ))
    for label in range(config.NUM_FACTORS):
        y = train_y[:, label]
        if y.min() == y.max() or int(y.sum()) < 3:
            probability[:, label] = valid_x[:, label, 0]
            continue
        model = LogisticRegression(
            C=float(c_value), solver="liblinear", max_iter=1000,
            class_weight="balanced", random_state=SEED,
        )
        model.fit(train_x[:, label, :], y)
        probability[:, label] = model.predict_proba(valid_x[:, label, :])[:, 1]
    return probability


def _choose_parameters(components, targets, indices, risk, groups):
    """Inner user-disjoint selection of one global C and prevalence ratio."""
    local_risk, local_groups = risk[indices], groups[indices]
    splitter = StratifiedGroupKFold(
        n_splits=4, shuffle=True, random_state=config.SEED + 808,
    )
    probability_by_c = {c: np.zeros((len(indices), config.NUM_FACTORS), np.float32) for c in CS}
    for fit_rel, valid_rel in splitter.split(np.zeros(len(indices)), local_risk, local_groups):
        fit_idx, valid_idx = indices[fit_rel], indices[valid_rel]
        fit_x, valid_x = _features(components, fit_idx), _features(components, valid_idx)
        for c_value in CS:
            probability_by_c[c_value][valid_rel] = _fit_predict(
                fit_x, targets[fit_idx], valid_x, c_value
            )
    prevalence = targets[indices].mean(0)
    rows = []
    for c_value, probability in probability_by_c.items():
        for ratio in RATIOS:
            prediction = _rank_decode(probability, prevalence, ratio)
            rows.append({
                "c": float(c_value), "ratio": float(ratio),
                "score": float(f1_score(
                    targets[indices], prediction, average="macro", zero_division=0
                )),
            })
    return max(rows, key=lambda row: row["score"]), rows


def cross_validate():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    components, targets = _components()
    frame = load_train_data().reset_index(drop=True)
    risk = frame.risk_label.to_numpy(np.int64)
    groups = frame.anon_user_id.astype(str).to_numpy()
    splitter = StratifiedGroupKFold(
        n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED,
    )
    baseline_oof = np.zeros_like(targets, dtype=bool)
    candidate_oof = np.zeros_like(targets, dtype=bool)
    candidate_probability = np.zeros_like(targets, dtype=np.float32)
    folds = []
    for fold, (train_idx, valid_idx) in enumerate(
        splitter.split(np.zeros(len(frame)), risk, groups)
    ):
        best, grid = _choose_parameters(
            components, targets, train_idx, risk, groups
        )
        train_x, valid_x = _features(components, train_idx), _features(components, valid_idx)
        meta = _fit_predict(train_x, targets[train_idx], valid_x, best["c"])
        prevalence = targets[train_idx].mean(0)
        candidate = _rank_decode(meta, prevalence, best["ratio"])
        baseline = _rank_decode(
            components["current_v3"][valid_idx], prevalence, 1.10
        )
        baseline_oof[valid_idx] = baseline
        candidate_oof[valid_idx] = candidate
        candidate_probability[valid_idx] = meta
        folds.append({
            "fold": fold, "train_rows": len(train_idx), "valid_rows": len(valid_idx),
            "c": best["c"], "ratio": best["ratio"],
            "inner_score": best["score"],
            "baseline_macro_f1": float(f1_score(
                targets[valid_idx], baseline, average="macro", zero_division=0
            )),
            "candidate_macro_f1": float(f1_score(
                targets[valid_idx], candidate, average="macro", zero_division=0
            )),
            "inner_top5": sorted(grid, key=lambda row: row["score"], reverse=True)[:5],
        })
        print(json.dumps(folds[-1], indent=2), flush=True)

    baseline_score = float(f1_score(
        targets, baseline_oof, average="macro", zero_division=0
    ))
    candidate_score = float(f1_score(
        targets, candidate_oof, average="macro", zero_division=0
    ))
    users = np.unique(groups); rng = np.random.default_rng(SEED); deltas = []
    for _ in range(3000):
        selected = rng.choice(users, size=len(users), replace=True)
        positions = np.concatenate([np.flatnonzero(groups == user) for user in selected])
        old = f1_score(targets[positions], baseline_oof[positions], average="macro", zero_division=0)
        new = f1_score(targets[positions], candidate_oof[positions], average="macro", zero_division=0)
        deltas.append(float(new - old))
    deltas = np.asarray(deltas)
    per_label = []
    for label in range(config.NUM_FACTORS):
        per_label.append({
            "label": config.ID2FACTOR[label], "support": int(targets[:, label].sum()),
            "baseline_f1": float(f1_score(
                targets[:, label], baseline_oof[:, label], zero_division=0
            )),
            "candidate_f1": float(f1_score(
                targets[:, label], candidate_oof[:, label], zero_division=0
            )),
        })
    bootstrap = {
        "mean_delta": float(deltas.mean()),
        "p05_delta": float(np.quantile(deltas, .05)),
        "p95_delta": float(np.quantile(deltas, .95)),
        "positive_fraction": float((deltas > 0).mean()),
    }
    adopted = bool(
        candidate_score >= baseline_score + .005
        and bootstrap["positive_fraction"] >= .80
    )
    payload = {
        "training_version": TRAINING_VERSION,
        "components": list(components), "folds": folds,
        "baseline_nested_macro_f1": baseline_score,
        "candidate_nested_macro_f1": candidate_score,
        "delta": candidate_score - baseline_score,
        "per_label": per_label, "user_cluster_bootstrap": bootstrap,
        "adopted": adopted,
    }
    np.savez_compressed(
        PREDICTION_FILE, probabilities=candidate_probability,
        predictions=candidate_oof, baseline_predictions=baseline_oof, targets=targets,
    )
    RESULTS_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    return payload


if __name__ == "__main__":
    cross_validate()
