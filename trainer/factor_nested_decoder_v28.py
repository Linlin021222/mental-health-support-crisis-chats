"""Nested per-label prevalence calibration for the accepted Task 2 ranking."""
from __future__ import annotations

import json

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

from configs.config import config
from inference.factor_nli import _rank_decode
from preprocess.preprocess import load_train_data
from trainer.factor_llm_lexical_v6 import _current_v3_probability
from trainer.factor_sentence_evidence_cv_v27 import _user_bootstrap


OUTPUT = config.OUTPUT_DIR / "factor_nested_decoder_v28"
RESULTS = OUTPUT / "cv_results.json"
CALIBRATION = OUTPUT / "calibration.json"
TRAINING_VERSION = "nested-shrunk-per-label-decoder-v28"
BASE_RATIO = 1.10
RATIO_GRID = np.asarray([
    .55, .65, .75, .85, .95, 1.05, 1.10, 1.15, 1.25, 1.35, 1.50,
], dtype=np.float32)


def _topk_label(score, count):
    count = max(1, min(len(score), int(count)))
    prediction = np.zeros(len(score), dtype=bool)
    selected = np.argpartition(score, len(score) - count)[len(score) - count:]
    prediction[selected] = True
    return prediction


def _select_ratios(probability, truth, prevalence):
    ratios, audit = np.full(config.NUM_FACTORS, BASE_RATIO, dtype=np.float32), []
    for label in range(config.NUM_FACTORS):
        support = int(truth[:, label].sum())
        rows = []
        for ratio in RATIO_GRID:
            count = int(round(len(truth) * float(prevalence[label]) * float(ratio)))
            prediction = _topk_label(probability[:, label], count)
            score = f1_score(truth[:, label], prediction, zero_division=0)
            rows.append((float(score), float(ratio)))
        best_score, best_ratio = max(
            rows, key=lambda row: (row[0], -abs(row[1] - BASE_RATIO))
        )
        # Empirical-Bayes shrinkage. With 10 positives only 20% of the raw
        # displacement survives; with 200 positives it retains 83%.
        reliability = support / (support + 40.0)
        shrunk = BASE_RATIO + reliability * (best_ratio - BASE_RATIO)
        shrunk = float(np.clip(shrunk, .70, 1.40))
        ratios[label] = shrunk
        audit.append({
            "label": config.ID2FACTOR[label], "support": support,
            "raw_best_ratio": best_ratio, "raw_best_f1": best_score,
            "reliability": reliability, "shrunk_ratio": shrunk,
        })
    return ratios, audit


def cross_validate():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame = load_train_data().reset_index(drop=True)
    targets = np.vstack(frame.factor_vector.to_numpy()).astype(np.int8)
    risk = frame.risk_label.to_numpy(dtype=np.int64)
    groups = frame.anon_user_id.astype(str).to_numpy()
    current, _ = _current_v3_probability()
    folds = list(StratifiedGroupKFold(
        n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED,
    ).split(np.zeros(len(frame)), risk, groups))

    baseline_prediction = np.zeros_like(targets, dtype=bool)
    candidate_prediction = np.zeros_like(targets, dtype=bool)
    fold_rows, fold_ratios = [], []
    for fold, (fit_idx, valid_idx) in enumerate(folds):
        prevalence = targets[fit_idx].mean(0)
        ratios, audit = _select_ratios(
            current[fit_idx], targets[fit_idx], prevalence
        )
        baseline_prediction[valid_idx] = _rank_decode(
            current[valid_idx], prevalence, BASE_RATIO
        )
        candidate_prediction[valid_idx] = _rank_decode(
            current[valid_idx], prevalence, ratios
        )
        baseline = f1_score(
            targets[valid_idx], baseline_prediction[valid_idx],
            average="macro", zero_division=0,
        )
        candidate = f1_score(
            targets[valid_idx], candidate_prediction[valid_idx],
            average="macro", zero_division=0,
        )
        fold_rows.append({
            "fold": fold, "baseline_macro_f1": float(baseline),
            "candidate_macro_f1": float(candidate),
            "delta": float(candidate - baseline), "selection": audit,
        })
        fold_ratios.append(ratios)

    baseline = float(f1_score(
        targets, baseline_prediction, average="macro", zero_division=0
    ))
    candidate = float(f1_score(
        targets, candidate_prediction, average="macro", zero_division=0
    ))
    bootstrap = _user_bootstrap(
        targets, baseline_prediction, candidate_prediction, groups,
        seed=282828, draws=3000,
    )
    production_ratios = np.median(np.stack(fold_ratios), axis=0).astype(np.float32)
    # This is diagnostic only; adoption is based on the nested predictions.
    production_baseline = _rank_decode(current, targets.mean(0), BASE_RATIO)
    production_prediction = _rank_decode(current, targets.mean(0), production_ratios)
    production_baseline_score = float(f1_score(
        targets, production_baseline, average="macro", zero_division=0
    ))
    production_score = float(f1_score(
        targets, production_prediction, average="macro", zero_division=0
    ))
    adopted = bool(
        candidate >= baseline + .005
        and bootstrap["positive_fraction"] >= .80
        and bootstrap["p05_delta"] >= 0
        and production_score >= production_baseline_score
    )
    per_label = [{
        "label": config.ID2FACTOR[label],
        "support": int(targets[:, label].sum()),
        "production_ratio": float(production_ratios[label]),
        "baseline_f1": float(f1_score(
            targets[:, label], baseline_prediction[:, label], zero_division=0
        )),
        "candidate_f1": float(f1_score(
            targets[:, label], candidate_prediction[:, label], zero_division=0
        )),
    } for label in range(config.NUM_FACTORS)]
    payload = {
        "training_version": TRAINING_VERSION,
        "evaluation": "five-fold nested user-disjoint; ratios selected on other-fold OOF only",
        "base_ratio": BASE_RATIO,
        "nested_baseline_macro_f1": baseline,
        "nested_candidate_macro_f1": candidate,
        "nested_delta": candidate - baseline,
        "production_baseline_oof_macro_f1": production_baseline_score,
        "production_candidate_oof_macro_f1": production_score,
        "production_delta": production_score - production_baseline_score,
        "user_cluster_bootstrap": bootstrap,
        "folds": fold_rows,
        "per_label": per_label,
        "adopted": adopted,
    }
    calibration = {
        "training_version": TRAINING_VERSION,
        "adopted": adopted,
        "ratios": {
            config.ID2FACTOR[label]: float(production_ratios[label])
            for label in range(config.NUM_FACTORS)
        },
        "nested_macro_f1": candidate,
        "nested_baseline_macro_f1": baseline,
        "bootstrap_positive_fraction": bootstrap["positive_fraction"],
    }
    RESULTS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    CALIBRATION.write_text(json.dumps(calibration, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    return payload


if __name__ == "__main__":
    cross_validate()
