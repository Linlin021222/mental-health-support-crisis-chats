"""Factor-balanced validation and stable label-wise routing for Task 2.

This is a CPU-only gate over already user-disjoint OOF probabilities.  It
addresses two failure modes found in the accepted pipeline:

1. folds stratified only by Task-1 risk can contain no positives for a rare
   factor;
2. one global prevalence ratio is shared by all 24 labels.

Every routing decision is selected on outer-fit users using a second set of
factor-balanced inner folds.  The outer validation users are never used for
parameter selection.
"""
from __future__ import annotations

import json
from collections import Counter

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

from configs.config import config
from preprocess.preprocess import load_train_data
from trainer.factor_sentence_evidence_cv_v27 import _user_bootstrap
from utils.multilabel_group_split import multilabel_group_folds, split_audit


OUTPUT = config.OUTPUT_DIR / "factor_balanced_calibration_v47"
RESULTS = OUTPUT / "results.json"
CALIBRATION = OUTPUT / "calibration.json"
TRAINING_VERSION = "factor-balanced-nested-routing-v47"
BASE_RATIO = 1.10
RATIO_GRID = (0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.30, 1.40)
WEIGHT_GRID = (0.15, 0.30, 0.45)
MIN_INNER_DELTA = 0.0075


def _rank(values: np.ndarray) -> np.ndarray:
    """Column-wise percentile ranks, making heterogeneous scores comparable."""
    values = np.asarray(values, dtype=np.float32)
    result = np.empty_like(values)
    denominator = max(1, len(values) - 1)
    for label in range(values.shape[1]):
        order = np.argsort(values[:, label], kind="stable")
        ranks = np.empty(len(values), dtype=np.float32)
        ranks[order] = np.arange(len(values), dtype=np.float32) / denominator
        result[:, label] = ranks
    return result


def _topk(score: np.ndarray, prevalence: float, ratio: float) -> np.ndarray:
    count = max(1, min(len(score), int(round(len(score) * prevalence * ratio))))
    selected = np.argpartition(score, len(score) - count)[len(score) - count:]
    prediction = np.zeros(len(score), dtype=bool)
    prediction[selected] = True
    return prediction


def _components() -> tuple[dict[str, np.ndarray], np.ndarray]:
    saved = np.load(config.OUTPUT_DIR / "factor_cv" / "factor_oof_predictions.npz")
    semantic = saved["semantic"].astype(np.float32)
    cpu = saved["cpu"].astype(np.float32)
    old = np.load(config.OUTPUT_DIR / "factor_cross_encoder" /
                  "oof_predictions.npz")["probabilities"].astype(np.float32)
    new = np.load(config.OUTPUT_DIR / "factor_cross_encoder_v2" /
                  "oof_predictions.npz")["probabilities"].astype(np.float32)
    calibration = json.loads((config.OUTPUT_DIR / "factor_cross_encoder_v2" /
                              "calibration.json").read_text(encoding="utf-8"))
    base = .70 * semantic + .30 * cpu
    current = (float(calibration["base_weight"]) * base
               + float(calibration["old_cross_weight"]) * old
               + float(calibration["new_cross_weight"]) * new)
    return {
        "current": current.astype(np.float32),
        "semantic": semantic,
        "cpu": cpu,
        "old_cross": old,
        "prototype_cross": new,
    }, saved["targets"].astype(np.int8)


def _inner_predictions(
    score: np.ndarray,
    truth: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    label: int,
    ratio: float,
) -> tuple[np.ndarray, list[float]]:
    prediction = np.zeros(len(truth), dtype=bool)
    fold_scores = []
    for inner_fit, inner_valid in folds:
        prevalence = float(truth[inner_fit, label].mean())
        local = _topk(score[inner_valid], prevalence, ratio)
        prediction[inner_valid] = local
        fold_scores.append(float(f1_score(
            truth[inner_valid, label], local, zero_division=0,
        )))
    return prediction, fold_scores


def _select_label(
    ranked: dict[str, np.ndarray],
    truth: np.ndarray,
    groups: np.ndarray,
    risk: np.ndarray,
    label: int,
    seed: int,
) -> dict:
    inner = multilabel_group_folds(
        truth, groups, risk, n_splits=4, seed=seed,
    )
    baseline_prediction, baseline_folds = _inner_predictions(
        ranked["current"][:, label], truth, inner, label, BASE_RATIO,
    )
    baseline = float(f1_score(
        truth[:, label], baseline_prediction, zero_division=0,
    ))
    candidates = []
    for expert in ("current", "semantic", "cpu", "old_cross", "prototype_cross"):
        weights = (0.0,) if expert == "current" else WEIGHT_GRID
        for weight in weights:
            score = ((1.0 - weight) * ranked["current"][:, label]
                     + weight * ranked[expert][:, label])
            for ratio in RATIO_GRID:
                prediction, fold_scores = _inner_predictions(
                    score, truth, inner, label, ratio,
                )
                value = float(f1_score(
                    truth[:, label], prediction, zero_division=0,
                ))
                std = float(np.std(fold_scores))
                positive_folds = int(sum(
                    new >= old - 1e-12
                    for new, old in zip(fold_scores, baseline_folds)
                ))
                # Prefer stable, simple changes when empirical F1 is tied.
                objective = (value - .025 * std - .004 * weight
                             - .002 * abs(ratio - BASE_RATIO))
                candidates.append({
                    "expert": expert,
                    "weight": float(weight),
                    "ratio": float(ratio),
                    "inner_f1": value,
                    "inner_std": std,
                    "positive_inner_folds": positive_folds,
                    "objective": objective,
                })
    selected = max(candidates, key=lambda row: row["objective"])
    # A label-specific route must improve pooled inner F1 and cannot rely on
    # only one lucky validation fold. Otherwise retain the accepted decoder.
    accepted = bool(
        selected["inner_f1"] >= baseline + MIN_INNER_DELTA
        and selected["positive_inner_folds"] >= 3
    )
    if not accepted:
        selected = {
            "expert": "current", "weight": 0.0, "ratio": BASE_RATIO,
            "inner_f1": baseline, "inner_std": float(np.std(baseline_folds)),
            "positive_inner_folds": 4, "objective": baseline,
        }
    selected.update({
        "label": config.ID2FACTOR[label],
        "support": int(truth[:, label].sum()),
        "baseline_inner_f1": baseline,
        "accepted": accepted,
    })
    return selected


def _decode_outer(
    components: dict[str, np.ndarray],
    targets: np.ndarray,
    frame,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    baseline = np.zeros_like(targets, dtype=bool)
    candidate = np.zeros_like(targets, dtype=bool)
    fold_rows = []
    groups = frame.anon_user_id.astype(str).to_numpy()
    risk = frame.risk_label.to_numpy(dtype=np.int64)
    for fold, (fit, valid) in enumerate(folds):
        fit_ranked = {name: _rank(value[fit]) for name, value in components.items()}
        valid_ranked = {name: _rank(value[valid]) for name, value in components.items()}
        selections = []
        for label in range(config.NUM_FACTORS):
            selected = _select_label(
                fit_ranked, targets[fit], groups[fit], risk[fit], label,
                seed=config.SEED + 4700 + fold * 31 + label,
            )
            prevalence = float(targets[fit, label].mean())
            baseline[valid, label] = _topk(
                valid_ranked["current"][:, label], prevalence, BASE_RATIO,
            )
            expert = selected["expert"]
            weight = selected["weight"]
            score = ((1.0 - weight) * valid_ranked["current"][:, label]
                     + weight * valid_ranked[expert][:, label])
            candidate[valid, label] = _topk(
                score, prevalence, selected["ratio"],
            )
            selections.append(selected)
        base_score = float(f1_score(
            targets[valid], baseline[valid], average="macro", zero_division=0,
        ))
        new_score = float(f1_score(
            targets[valid], candidate[valid], average="macro", zero_division=0,
        ))
        fold_rows.append({
            "fold": fold,
            "baseline_macro_f1": base_score,
            "candidate_macro_f1": new_score,
            "delta": new_score - base_score,
            "selections": selections,
        })
        print(f"V47 fold={fold} Task2 {base_score:.6f} -> {new_score:.6f} "
              f"({new_score-base_score:+.6f})", flush=True)
    return baseline, candidate, fold_rows


def _per_label(targets, baseline, candidate):
    rows = []
    for label in range(config.NUM_FACTORS):
        old = float(f1_score(targets[:, label], baseline[:, label], zero_division=0))
        new = float(f1_score(targets[:, label], candidate[:, label], zero_division=0))
        rows.append({
            "label": config.ID2FACTOR[label],
            "support": int(targets[:, label].sum()),
            "baseline_f1": old, "candidate_f1": new, "delta": new - old,
        })
    return rows


def _production_parameters(fold_rows):
    result = []
    for label in range(config.NUM_FACTORS):
        selections = [row["selections"][label] for row in fold_rows]
        expert = Counter(x["expert"] for x in selections).most_common(1)[0][0]
        matching = [x for x in selections if x["expert"] == expert]
        result.append({
            "label": config.ID2FACTOR[label],
            "expert": expert,
            "weight": float(np.median([x["weight"] for x in matching])),
            "ratio": float(np.median([x["ratio"] for x in matching])),
            "outer_acceptance_count": int(sum(x["accepted"] for x in selections)),
        })
    return result


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame = load_train_data().reset_index(drop=True)
    components, targets = _components()
    groups = frame.anon_user_id.astype(str).to_numpy()
    risk = frame.risk_label.to_numpy(dtype=np.int64)

    old_folds = list(StratifiedGroupKFold(
        n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED,
    ).split(np.zeros(len(frame)), risk, groups))
    balanced_folds = multilabel_group_folds(
        targets, groups, risk, n_splits=config.N_FOLDS, seed=config.SEED + 47,
    )
    old_audit = split_audit(old_folds, targets, groups)
    balanced_audit = split_audit(balanced_folds, targets, groups)
    print("V47 split audit: "
          f"zero factor-folds {old_audit['total_zero_positive_fold_labels']} -> "
          f"{balanced_audit['total_zero_positive_fold_labels']}; "
          f"mean support CV {old_audit['mean_factor_support_cv']:.4f} -> "
          f"{balanced_audit['mean_factor_support_cv']:.4f}", flush=True)

    baseline, candidate, fold_rows = _decode_outer(
        components, targets, frame, balanced_folds,
    )
    baseline_score = float(f1_score(
        targets, baseline, average="macro", zero_division=0,
    ))
    candidate_score = float(f1_score(
        targets, candidate, average="macro", zero_division=0,
    ))
    bootstrap = _user_bootstrap(
        targets, baseline, candidate, groups, seed=474747, draws=4000,
    )
    per_label = _per_label(targets, baseline, candidate)
    weak = [row for row in per_label if row["baseline_f1"] < .60]
    weak_old = float(np.mean([row["baseline_f1"] for row in weak]))
    weak_new = float(np.mean([row["candidate_f1"] for row in weak]))
    adopted = bool(
        candidate_score >= baseline_score + .005
        and bootstrap["positive_fraction"] >= .80
        and bootstrap["p05_delta"] >= 0
        and weak_new >= weak_old
    )
    production = _production_parameters(fold_rows)
    payload = {
        "training_version": TRAINING_VERSION,
        "evaluation": "five outer factor-balanced user-disjoint folds; four inner factor-balanced user-disjoint folds",
        "split_audit": {"old_risk_stratified": old_audit, "balanced": balanced_audit},
        "baseline_macro_f1": baseline_score,
        "candidate_macro_f1": candidate_score,
        "delta": candidate_score - baseline_score,
        "weak_label_mean_baseline_f1": weak_old,
        "weak_label_mean_candidate_f1": weak_new,
        "user_cluster_bootstrap": bootstrap,
        "folds": fold_rows,
        "per_label": per_label,
        "production_parameters": production,
        "adopted": adopted,
    }
    calibration = {
        "training_version": TRAINING_VERSION,
        "adopted": adopted,
        "nested_macro_f1": candidate_score,
        "nested_baseline_macro_f1": baseline_score,
        "parameters": production,
    }
    RESULTS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    CALIBRATION.write_text(json.dumps(calibration, indent=2), encoding="utf-8")
    print(json.dumps({
        "training_version": TRAINING_VERSION,
        "baseline_macro_f1": baseline_score,
        "candidate_macro_f1": candidate_score,
        "delta": candidate_score - baseline_score,
        "weak_label_mean_delta": weak_new - weak_old,
        "bootstrap": bootstrap,
        "adopted": adopted,
    }, indent=2), flush=True)
    return payload


if __name__ == "__main__":
    main()
