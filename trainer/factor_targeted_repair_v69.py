"""Strict label-local repair for the weakest Task-2 factors.

The official V68 result showed that replacing the shared sparse branch can
damage otherwise healthy labels.  V69 therefore diagnoses all 24 labels but
changes only two pre-specified weak labels whose former nested experiment
selected the same kind of correction in every outer fold:

* sexual orientation related issues: retain the accepted ensemble ranking and
  shrink only its predicted prevalence;
* sense of responsibility: add a small MentalRoBERTa semantic residual and use
  a label-specific prevalence ratio.

Every other factor column is copied bit-for-bit from the accepted baseline.
"""
from __future__ import annotations

import json

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from configs.config import config
from preprocess.preprocess import load_train_data
from trainer.factor_balanced_calibration_v47 import (
    _components,
    _decode_outer,
    _production_parameters,
)
from trainer.factor_sentence_evidence_cv_v27 import _user_bootstrap
from utils.multilabel_group_split import multilabel_group_folds


OUTPUT = config.OUTPUT_DIR / "factor_targeted_repair_v69"
RESULTS = OUTPUT / "results.json"
CALIBRATION = OUTPUT / "calibration.json"
TRAINING_VERSION = "label-local-targeted-repair-v69"
TARGET_NAMES = (
    "sexual orientation related issues",
    "sense of responsibility",
)


def _safe_auc(metric, truth, score):
    if np.unique(truth).size < 2:
        return None
    return float(metric(truth, score))


def _diagnostics(targets, probability, prediction, folds):
    rows = []
    for label in range(config.NUM_FACTORS):
        truth = targets[:, label].astype(bool)
        pred = prediction[:, label].astype(bool)
        fold_f1 = [
            float(f1_score(targets[valid, label], prediction[valid, label],
                           zero_division=0))
            for _, valid in folds
        ]
        tp = int(np.sum(truth & pred)); fp = int(np.sum(~truth & pred))
        fn = int(np.sum(truth & ~pred)); tn = int(np.sum(~truth & ~pred))
        f1 = float(f1_score(truth, pred, zero_division=0))
        auc = _safe_auc(roc_auc_score, truth, probability[:, label])
        pr_auc = _safe_auc(average_precision_score, truth, probability[:, label])
        if f1 < .60 and auc is not None and auc >= .92:
            bottleneck = "decoder/calibration"
        elif f1 < .60:
            bottleneck = "representation/ranking"
        else:
            bottleneck = "not currently weak"
        rows.append({
            "label": config.ID2FACTOR[label], "support": int(truth.sum()),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": float(precision_score(truth, pred, zero_division=0)),
            "recall": float(recall_score(truth, pred, zero_division=0)),
            "f1": f1, "roc_auc": auc, "pr_auc": pr_auc,
            "fold_f1": fold_f1, "fold_f1_std": float(np.std(fold_f1)),
            "bottleneck": bottleneck,
        })
    return rows


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame = load_train_data().reset_index(drop=True)
    components, targets = _components()
    groups = frame.anon_user_id.astype(str).to_numpy()
    risk = frame.risk_label.to_numpy(np.int64)
    folds = multilabel_group_folds(
        targets, groups, risk, n_splits=config.N_FOLDS, seed=config.SEED + 47,
    )

    # Reuse V47's genuinely nested choices, but discard all non-target labels.
    baseline, routed, fold_rows = _decode_outer(components, targets, frame, folds)
    target_ids = [config.FACTOR2ID[name] for name in TARGET_NAMES]
    candidate = baseline.copy()
    candidate[:, target_ids] = routed[:, target_ids]

    base_score = float(f1_score(targets, baseline, average="macro", zero_division=0))
    new_score = float(f1_score(targets, candidate, average="macro", zero_division=0))
    per_label = []
    for label in range(config.NUM_FACTORS):
        old = float(f1_score(targets[:, label], baseline[:, label], zero_division=0))
        new = float(f1_score(targets[:, label], candidate[:, label], zero_division=0))
        per_label.append({
            "label": config.ID2FACTOR[label], "support": int(targets[:, label].sum()),
            "baseline_f1": old, "candidate_f1": new, "delta": new-old,
            "changed": bool(label in target_ids),
        })
    fold_metrics = []
    for fold, (_, valid) in enumerate(folds):
        old = float(f1_score(targets[valid], baseline[valid], average="macro", zero_division=0))
        new = float(f1_score(targets[valid], candidate[valid], average="macro", zero_division=0))
        fold_metrics.append({"fold": fold, "baseline": old, "candidate": new,
                             "delta": new-old})
    bootstrap = _user_bootstrap(
        targets, baseline, candidate, groups, seed=696969, draws=5000,
    )

    production = _production_parameters(fold_rows)
    production = [row for row in production if row["label"] in TARGET_NAMES]
    target_deltas = [per_label[label]["delta"] for label in target_ids]
    acceptance_counts = [row["outer_acceptance_count"] for row in production]
    experimental_adopted = bool(
        new_score >= base_score + .003
        and all(delta > 0 for delta in target_deltas)
        and all(count >= 4 for count in acceptance_counts)
        and sum(row["delta"] >= -1e-12 for row in fold_metrics) >= 4
        and bootstrap["positive_fraction"] >= .80
    )
    diagnostics = _diagnostics(targets, components["current"], baseline, folds)
    payload = {
        "training_version": TRAINING_VERSION,
        "evaluation": "nested user-disjoint routing; exactly two target labels may change",
        "target_labels": list(TARGET_NAMES),
        "baseline_macro_f1": base_score,
        "candidate_macro_f1": new_score,
        "delta": new_score-base_score,
        "folds": fold_metrics,
        "bootstrap": bootstrap,
        "per_label": per_label,
        "baseline_diagnostics": diagnostics,
        "production_parameters": production,
        "experimental_adopted": experimental_adopted,
        "production_adopted": False,
    }
    calibration = {
        "training_version": TRAINING_VERSION,
        "experimental_adopted": experimental_adopted,
        "production_adopted": False,
        "training_prevalence": targets.mean(0).tolist(),
        "parameters": production,
    }
    RESULTS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    CALIBRATION.write_text(json.dumps(calibration, indent=2), encoding="utf-8")
    print(json.dumps({
        "training_version": TRAINING_VERSION,
        "baseline_macro_f1": base_score,
        "candidate_macro_f1": new_score,
        "delta": new_score-base_score,
        "target_results": [per_label[label] for label in target_ids],
        "folds": fold_metrics,
        "bootstrap": bootstrap,
        "production_parameters": production,
        "experimental_adopted": experimental_adopted,
    }, indent=2), flush=True)
    return payload


if __name__ == "__main__":
    main()
