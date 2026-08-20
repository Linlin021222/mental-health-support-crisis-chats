"""Five-fold user-disjoint sparse retrieval expert for Task 2 tail labels."""
from __future__ import annotations

import json

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import normalize
from tqdm import tqdm

from baseline import _vectorizer
from configs.config import config
from preprocess.preprocess import load_train_data
from utils.factor_calibration import apply_prior_topk


OUTPUT_DIR = config.OUTPUT_DIR / "factor_knn_v5"
OOF_FILE = OUTPUT_DIR / "oof_predictions.npz"
RESULT_FILE = OUTPUT_DIR / "cv_results.json"
CALIBRATION_FILE = OUTPUT_DIR / "calibration.json"
TRAINING_VERSION = "sparse-positive-retrieval-v5"


def _top_mean(values, count):
    count = min(int(count), values.shape[1])
    if count <= 0:
        return np.zeros(values.shape[0], dtype=np.float32)
    indices = np.argpartition(values, values.shape[1] - count, axis=1)[:, -count:]
    return np.take_along_axis(values, indices, axis=1).mean(1)


def _retrieval_scores(similarity, train_targets):
    n = similarity.shape[0]
    result = {
        "positive_max": np.zeros((n, config.NUM_FACTORS), dtype=np.float32),
        "positive_top3": np.zeros((n, config.NUM_FACTORS), dtype=np.float32),
        "positive_top5": np.zeros((n, config.NUM_FACTORS), dtype=np.float32),
        "positive_margin": np.zeros((n, config.NUM_FACTORS), dtype=np.float32),
    }
    for label in range(config.NUM_FACTORS):
        positive = np.flatnonzero(train_targets[:, label] > 0)
        negative = np.flatnonzero(train_targets[:, label] == 0)
        if len(positive):
            values = similarity[:, positive]
            result["positive_max"][:, label] = values.max(1)
            result["positive_top3"][:, label] = _top_mean(values, 3)
            result["positive_top5"][:, label] = _top_mean(values, 5)
            negative_max = similarity[:, negative].max(1) if len(negative) else 0.0
            result["positive_margin"][:, label] = values.max(1) - negative_max
    for neighbours in (10, 20, 40):
        count = min(neighbours, similarity.shape[1])
        indices = np.argpartition(
            similarity, similarity.shape[1] - count, axis=1
        )[:, -count:]
        values = np.take_along_axis(similarity, indices, axis=1)
        weights = np.maximum(values, 0.0) ** 3
        target = train_targets[indices]
        result[f"knn_{neighbours}"] = (
            (weights[..., None] * target).sum(1)
            / weights.sum(1, keepdims=True).clip(min=1e-8)
        ).astype(np.float32)
    return result


def _macro(targets, probabilities, prevalence, ratio):
    return float(f1_score(
        targets, apply_prior_topk(probabilities, prevalence, ratio),
        average="macro", zero_division=0,
    ))


def _grid(current, variants, targets, indices, prevalence):
    rows = []
    for name, retrieval in variants.items():
        for weight in (0.0, 0.05, 0.10, 0.20, 0.30, 0.40):
            probability = (1.0 - weight) * current[indices] + weight * retrieval[indices]
            for ratio in (1.0, 1.10, 1.25):
                rows.append({
                    "variant": name, "retrieval_weight": weight,
                    "prevalence_ratio": ratio,
                    "macro_f1": _macro(targets[indices], probability, prevalence, ratio),
                })
    return sorted(rows, key=lambda item: item["macro_f1"], reverse=True)


def main(force=False):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = load_train_data().reset_index(drop=True)
    targets = np.vstack(frame.factor_vector.to_numpy()).astype(np.int8)
    folds = list(StratifiedGroupKFold(
        n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED
    ).split(np.zeros(len(frame)), frame.risk_label, frame.anon_user_id))

    variants = None
    if OOF_FILE.exists() and not force:
        saved = np.load(OOF_FILE)
        variants = {key: saved[key] for key in saved.files if key != "targets"}
        if not np.array_equal(saved["targets"], targets):
            variants = None
    if variants is None:
        names = ["positive_max", "positive_top3", "positive_top5", "positive_margin",
                 "knn_10", "knn_20", "knn_40"]
        variants = {name: np.zeros_like(targets, dtype=np.float32) for name in names}
        for fold, (train_idx, valid_idx) in enumerate(tqdm(folds, desc="retrieval OOF")):
            vectorizer = _vectorizer()
            train_matrix = normalize(vectorizer.fit_transform(frame.text.iloc[train_idx]))
            valid_matrix = normalize(vectorizer.transform(frame.text.iloc[valid_idx]))
            similarity = (valid_matrix @ train_matrix.T).toarray().astype(np.float32)
            scores = _retrieval_scores(similarity, targets[train_idx])
            for name in names:
                variants[name][valid_idx] = scores[name]
            print(f"retrieval fold {fold}: train={len(train_idx)} valid={len(valid_idx)}")
        np.savez_compressed(OOF_FILE, **variants, targets=targets)

    base_saved = np.load(config.OUTPUT_DIR / "factor_cv" / "factor_oof_predictions.npz")
    base = (
        config.FACTOR_SEMANTIC_MODEL_WEIGHT * base_saved["semantic"]
        + config.FACTOR_CPU_ENSEMBLE_WEIGHT * base_saved["cpu"]
    )
    old_cross = np.load(
        config.OUTPUT_DIR / "factor_cross_encoder" / "oof_predictions.npz"
    )["probabilities"]
    v3 = np.load(
        config.OUTPUT_DIR / "factor_cross_encoder_v2" / "oof_predictions.npz"
    )["probabilities"]
    v3_calibration = json.loads((
        config.OUTPUT_DIR / "factor_cross_encoder_v2" / "calibration.json"
    ).read_text(encoding="utf-8"))
    current = (
        float(v3_calibration["base_weight"]) * base
        + float(v3_calibration["old_cross_weight"]) * old_cross
        + float(v3_calibration["new_cross_weight"]) * v3
    )

    full_grid = _grid(
        current, variants, targets, np.arange(len(targets)), targets.mean(0)
    )
    nested_prediction = np.zeros_like(targets, dtype=bool)
    baseline_prediction = np.zeros_like(targets, dtype=bool)
    parameters = []
    for fold, (fit, valid) in enumerate(folds):
        selected = _grid(current, variants, targets, fit, targets[fit].mean(0))[0]
        probability = (
            (1.0 - selected["retrieval_weight"]) * current[valid]
            + selected["retrieval_weight"] * variants[selected["variant"]][valid]
        )
        nested_prediction[valid] = apply_prior_topk(
            probability, targets[fit].mean(0), selected["prevalence_ratio"]
        )
        baseline_prediction[valid] = apply_prior_topk(
            current[valid], targets[fit].mean(0),
            float(v3_calibration["prevalence_ratio"]),
        )
        parameters.append({"fold": fold, **selected})
    nested_score = float(f1_score(
        targets, nested_prediction, average="macro", zero_division=0
    ))
    baseline_nested = float(f1_score(
        targets, baseline_prediction, average="macro", zero_division=0
    ))

    # A production rule must be stable across folds: use the mode retrieval
    # family and median continuous/discrete parameters, never the full-OOF max.
    variant_counts = {}
    for item in parameters:
        variant_counts[item["variant"]] = variant_counts.get(item["variant"], 0) + 1
    production_variant = max(variant_counts, key=variant_counts.get)
    production_weight = float(np.median([
        item["retrieval_weight"] for item in parameters
        if item["variant"] == production_variant
    ]))
    production_ratio = float(np.median([
        item["prevalence_ratio"] for item in parameters
    ]))
    production_probability = (
        (1.0 - production_weight) * current
        + production_weight * variants[production_variant]
    )
    production_score = _macro(
        targets, production_probability, targets.mean(0), production_ratio
    )
    baseline_same_ratio = _macro(targets, current, targets.mean(0), production_ratio)
    baseline_fixed = _macro(
        targets, current, targets.mean(0), float(v3_calibration["prevalence_ratio"])
    )
    adopted = bool(
        production_weight > 0
        and nested_score >= baseline_nested + 0.003
        and production_score >= baseline_same_ratio + 0.002
        and production_score >= baseline_fixed
    )
    calibration = {
        "training_version": TRAINING_VERSION, "adopted": adopted,
        "variant": production_variant, "retrieval_weight": production_weight,
        "existing_weight": 1.0 - production_weight,
        "prevalence_ratio": production_ratio,
        "nested_macro_f1": nested_score,
        "baseline_nested_macro_f1": baseline_nested,
        "production_oof_macro_f1": production_score,
        "baseline_same_ratio_oof_macro_f1": baseline_same_ratio,
        "baseline_fixed_oof_macro_f1": baseline_fixed,
    }
    result = {
        "training_version": TRAINING_VERSION, "calibration": calibration,
        "crossfit_parameters": parameters,
        "best_full_oof_optimistic": full_grid[0], "top15": full_grid[:15],
    }
    RESULT_FILE.write_text(json.dumps(result, indent=2), encoding="utf-8")
    CALIBRATION_FILE.write_text(json.dumps(calibration, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2)); return result


if __name__ == "__main__":
    main()
