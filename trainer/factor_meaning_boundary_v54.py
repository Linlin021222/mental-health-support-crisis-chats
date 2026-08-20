"""Evaluate the graded meaning-in-life boundary policy on balanced folds."""
from __future__ import annotations

import json

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from configs.config import config
from inference.factor_boundary_lexicon_v50 import boundary_flags
from inference.factor_meaning_boundary_v54 import strong_meaning_flags
from inference.factor_nli import _rank_decode
from preprocess.preprocess import load_train_data
from trainer.factor_rare_semantic_v49 import _baseline_probability
from trainer.factor_sentence_evidence_cv_v27 import _user_bootstrap
from utils.multilabel_group_split import multilabel_group_folds


OUTPUT = config.OUTPUT_DIR / "factor_meaning_boundary_v54"
RESULTS = OUTPUT / "results.json"
CALIBRATION = OUTPUT / "calibration.json"
LABEL = 23
ADDITIONAL_BROAD_BOOST = .10
STRONG_BOOST = .40
TRAINING_VERSION = "graded-meaning-boundary-v54"


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame = load_train_data().reset_index(drop=True)
    targets = np.vstack(frame.factor_vector.to_numpy()).astype(np.int8)
    groups = frame.anon_user_id.astype(str).to_numpy()
    risk = frame.risk_label.to_numpy(dtype=np.int64)
    folds = multilabel_group_folds(targets, groups, risk, 5, config.SEED + 47)
    texts = frame.text.astype(str).tolist()
    flags = boundary_flags(texts); strong = strong_meaning_flags(texts)
    base = _baseline_probability()
    v50 = base.copy(); v50[:, 13] += .10 * flags[:, 13]
    v50[:, 18] += .50 * flags[:, 18]; v50[:, LABEL] += .20 * flags[:, LABEL]
    candidate = v50.copy()
    candidate[:, LABEL] += ADDITIONAL_BROAD_BOOST * flags[:, LABEL]
    candidate[:, LABEL] += STRONG_BOOST * strong
    old_prediction = np.zeros_like(targets, dtype=bool)
    new_prediction = np.zeros_like(targets, dtype=bool)
    fold_rows = []
    for fold, (fit, valid) in enumerate(folds):
        prevalence = targets[fit].mean(0)
        old_prediction[valid] = _rank_decode(v50[valid], prevalence, 1.10)
        new_prediction[valid] = _rank_decode(candidate[valid], prevalence, 1.10)
        old = float(f1_score(targets[valid], old_prediction[valid], average="macro", zero_division=0))
        new = float(f1_score(targets[valid], new_prediction[valid], average="macro", zero_division=0))
        fold_rows.append({
            "fold": fold, "baseline_macro_f1": old,
            "candidate_macro_f1": new, "delta": new-old,
            "baseline_meaning_f1": float(f1_score(
                targets[valid, LABEL], old_prediction[valid, LABEL], zero_division=0)),
            "candidate_meaning_f1": float(f1_score(
                targets[valid, LABEL], new_prediction[valid, LABEL], zero_division=0)),
        })
    old = float(f1_score(targets, old_prediction, average="macro", zero_division=0))
    new = float(f1_score(targets, new_prediction, average="macro", zero_division=0))
    old_meaning = float(f1_score(targets[:, LABEL], old_prediction[:, LABEL], zero_division=0))
    new_meaning = float(f1_score(targets[:, LABEL], new_prediction[:, LABEL], zero_division=0))
    bootstrap = _user_bootstrap(targets, old_prediction, new_prediction, groups,
                                seed=545454, draws=4000)
    adopted = bool(
        new >= old + .0015 and new_meaning > old_meaning
        and all(row["delta"] >= -1e-12 for row in fold_rows)
        and bootstrap["positive_fraction"] >= .70
    )
    payload = {
        "training_version": TRAINING_VERSION,
        "evaluation": "fixed graded boundary; five factor-balanced user-disjoint folds",
        "baseline_macro_f1": old, "candidate_macro_f1": new, "delta": new-old,
        "baseline_meaning_f1": old_meaning, "candidate_meaning_f1": new_meaning,
        "meaning_delta": new_meaning-old_meaning,
        "meaning_roc_auc": float(roc_auc_score(targets[:, LABEL], candidate[:, LABEL])),
        "meaning_pr_auc": float(average_precision_score(targets[:, LABEL], candidate[:, LABEL])),
        "broad_matches": int(flags[:, LABEL].sum()), "strong_matches": int(strong.sum()),
        "strong_true_positives": int((strong * targets[:, LABEL]).sum()),
        "folds": fold_rows, "user_cluster_bootstrap": bootstrap,
        "adopted": adopted,
        "caveat": "Definition-derived policy was development-audited; confirm on leaderboard.",
    }
    RESULTS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    CALIBRATION.write_text(json.dumps({
        "training_version": TRAINING_VERSION, "adopted": adopted,
        "additional_broad_boost": ADDITIONAL_BROAD_BOOST,
        "strong_boost": STRONG_BOOST,
        "nested_macro_f1": new, "nested_baseline_macro_f1": old,
    }, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    return payload


if __name__ == "__main__":
    main()
