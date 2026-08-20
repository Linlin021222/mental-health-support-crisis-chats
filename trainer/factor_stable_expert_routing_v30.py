"""Nested stable per-label routing between V3 and its prototype expert."""
from __future__ import annotations

import json

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from configs.config import config
from inference.factor_nli import _rank_decode
from preprocess.preprocess import load_train_data
from trainer.factor_llm_lexical_v6 import _current_v3_probability
from trainer.factor_sentence_evidence_cv_v27 import _user_bootstrap


OUTPUT = config.OUTPUT_DIR / "factor_stable_expert_routing_v30"
RESULTS = OUTPUT / "cv_results.json"
CALIBRATION = OUTPUT / "calibration.json"
TRAINING_VERSION = "nested-stable-prototype-routing-v30"
WEIGHTS = (0.0, 0.10, 0.20, 0.30, 0.40)
RATIO = 1.10


def _label_prediction(score, prevalence, ratio=RATIO):
    count = max(1, min(len(score), int(round(len(score) * prevalence * ratio))))
    prediction = np.zeros(len(score), dtype=bool)
    prediction[np.argpartition(score, len(score) - count)[len(score) - count:]] = True
    return prediction


def _select_label_weights(current, prototype, targets, fit_idx, component_folds):
    weights = np.zeros(config.NUM_FACTORS, dtype=np.float32)
    audit = []
    prevalence = targets[fit_idx].mean(0)
    for label in range(config.NUM_FACTORS):
        truth = targets[fit_idx, label]
        baseline_prediction = _label_prediction(
            current[fit_idx, label], prevalence[label]
        )
        baseline = f1_score(truth, baseline_prediction, zero_division=0)
        block_baseline = []
        for block in component_folds:
            block_prevalence = targets[np.setdiff1d(fit_idx, block), label].mean()
            block_baseline.append(f1_score(
                targets[block, label],
                _label_prediction(current[block, label], block_prevalence),
                zero_division=0,
            ))
        candidates = []
        for weight in WEIGHTS:
            mixed = (
                (1.0 - weight) * current[:, label]
                + weight * prototype[:, label]
            )
            total = f1_score(
                truth, _label_prediction(mixed[fit_idx], prevalence[label]),
                zero_division=0,
            )
            block_values = []
            for block in component_folds:
                block_prevalence = targets[np.setdiff1d(fit_idx, block), label].mean()
                block_values.append(f1_score(
                    targets[block, label],
                    _label_prediction(mixed[block], block_prevalence),
                    zero_division=0,
                ))
            block_values = np.asarray(block_values)
            positive = int((block_values > np.asarray(block_baseline) + 1e-12).sum())
            negative = int((block_values < np.asarray(block_baseline) - 1e-12).sum())
            candidates.append((float(total), positive, -negative, float(weight)))
        best, positive, negative_signed, weight = max(
            candidates, key=lambda row: (row[0], row[1], row[2], -row[3])
        )
        support = int(truth.sum())
        required = 0.06 if support < 20 else (0.025 if support < 80 else 0.012)
        # Must improve globally and on at least half of the other outer folds,
        # without losing on more than one. This is much stricter than choosing
        # on the pooled OOF labels alone.
        negative = -negative_signed
        if best < baseline + required or positive < 2 or negative > 1:
            weight = 0.0
            best = float(baseline)
        weights[label] = weight
        audit.append({
            "label": config.ID2FACTOR[label], "support": support,
            "baseline_f1": float(baseline), "selected_f1": float(best),
            "positive_component_folds": positive,
            "negative_component_folds": negative,
            "selected_weight_toward_prototype": weight,
        })
    return weights, audit


def cross_validate():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame = load_train_data().reset_index(drop=True)
    targets = np.vstack(frame.factor_vector.to_numpy()).astype(np.int8)
    risk = frame.risk_label.to_numpy(dtype=np.int64)
    groups = frame.anon_user_id.astype(str).to_numpy()
    folds = list(StratifiedGroupKFold(
        n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED,
    ).split(np.zeros(len(frame)), risk, groups))
    current, _ = _current_v3_probability()
    prototype = np.load(
        config.OUTPUT_DIR / "factor_cross_encoder_v2" / "oof_predictions.npz"
    )["probabilities"].astype(np.float32)

    baseline_prediction = np.zeros_like(targets, dtype=bool)
    candidate_prediction = np.zeros_like(targets, dtype=bool)
    fold_weights, rows = [], []
    for outer, (fit_idx, valid_idx) in enumerate(folds):
        component_folds = [folds[i][1] for i in range(config.N_FOLDS) if i != outer]
        weights, audit = _select_label_weights(
            current, prototype, targets, fit_idx, component_folds
        )
        mixed = (
            (1.0 - weights[None, :]) * current[valid_idx]
            + weights[None, :] * prototype[valid_idx]
        )
        prevalence = targets[fit_idx].mean(0)
        baseline_prediction[valid_idx] = _rank_decode(
            current[valid_idx], prevalence, RATIO
        )
        candidate_prediction[valid_idx] = _rank_decode(mixed, prevalence, RATIO)
        baseline = f1_score(
            targets[valid_idx], baseline_prediction[valid_idx],
            average="macro", zero_division=0,
        )
        candidate = f1_score(
            targets[valid_idx], candidate_prediction[valid_idx],
            average="macro", zero_division=0,
        )
        rows.append({
            "fold": outer, "selected_nonzero_labels": int((weights > 0).sum()),
            "baseline_macro_f1": float(baseline),
            "candidate_macro_f1": float(candidate),
            "delta": float(candidate - baseline), "selection": audit,
        })
        fold_weights.append(weights)

    baseline = float(f1_score(
        targets, baseline_prediction, average="macro", zero_division=0
    ))
    candidate = float(f1_score(
        targets, candidate_prediction, average="macro", zero_division=0
    ))
    bootstrap = _user_bootstrap(
        targets, baseline_prediction, candidate_prediction, groups,
        seed=303030, draws=3000,
    )
    production_weights = np.median(np.stack(fold_weights), axis=0)
    production_mixed = (
        (1.0 - production_weights[None, :]) * current
        + production_weights[None, :] * prototype
    )
    production_baseline_prediction = _rank_decode(
        current, targets.mean(0), RATIO
    )
    production_prediction = _rank_decode(
        production_mixed, targets.mean(0), RATIO
    )
    production_baseline = float(f1_score(
        targets, production_baseline_prediction, average="macro", zero_division=0
    ))
    production = float(f1_score(
        targets, production_prediction, average="macro", zero_division=0
    ))
    adopted = bool(
        candidate >= baseline + .005
        and bootstrap["positive_fraction"] >= .80
        and bootstrap["p05_delta"] >= 0
        and production >= production_baseline
    )
    per_label = [{
        "label": config.ID2FACTOR[label],
        "support": int(targets[:, label].sum()),
        "production_weight_toward_prototype": float(production_weights[label]),
        "baseline_f1": float(f1_score(
            targets[:, label], baseline_prediction[:, label], zero_division=0
        )),
        "candidate_f1": float(f1_score(
            targets[:, label], candidate_prediction[:, label], zero_division=0
        )),
    } for label in range(config.NUM_FACTORS)]
    payload = {
        "training_version": TRAINING_VERSION,
        "evaluation": "nested user-disjoint; label routing selected on other four OOF folds",
        "ratio": RATIO,
        "nested_baseline_macro_f1": baseline,
        "nested_candidate_macro_f1": candidate,
        "nested_delta": candidate - baseline,
        "production_baseline_oof_macro_f1": production_baseline,
        "production_candidate_oof_macro_f1": production,
        "production_delta": production - production_baseline,
        "production_nonzero_labels": int((production_weights > 0).sum()),
        "user_cluster_bootstrap": bootstrap,
        "folds": rows, "per_label": per_label,
        "adopted": adopted,
    }
    calibration = {
        "training_version": TRAINING_VERSION, "adopted": adopted,
        "weights": {
            config.ID2FACTOR[label]: float(production_weights[label])
            for label in range(config.NUM_FACTORS)
        },
        "nested_macro_f1": candidate,
        "nested_baseline_macro_f1": baseline,
    }
    RESULTS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    CALIBRATION.write_text(json.dumps(calibration, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    return payload


if __name__ == "__main__":
    cross_validate()
