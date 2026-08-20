"""CodiEsp-style definition gazetteer gate for semantically rare factors."""
from __future__ import annotations

import json

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from configs.config import config
from inference.factor_boundary_lexicon_v50 import DEFAULT_BOOSTS, boundary_flags
from inference.factor_nli import _rank_decode
from preprocess.preprocess import load_train_data
from trainer.factor_rare_semantic_v49 import _baseline_probability
from trainer.factor_sentence_evidence_cv_v27 import _user_bootstrap
from utils.multilabel_group_split import multilabel_group_folds


OUTPUT = config.OUTPUT_DIR / "factor_boundary_lexicon_v50"
RESULTS = OUTPUT / "results.json"
CALIBRATION = OUTPUT / "calibration.json"
TRAINING_VERSION = "definition-boundary-gazetteer-v50"


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame = load_train_data().reset_index(drop=True)
    targets = np.vstack(frame.factor_vector.to_numpy()).astype(np.int8)
    groups = frame.anon_user_id.astype(str).to_numpy()
    risk = frame.risk_label.to_numpy(dtype=np.int64)
    folds = multilabel_group_folds(targets, groups, risk, 5, config.SEED + 47)
    probability = _baseline_probability()
    flags = boundary_flags(frame.text.astype(str).tolist())
    adjusted = probability.copy()
    for label, boost in DEFAULT_BOOSTS.items():
        adjusted[:, label] += boost * flags[:, label]
    baseline = np.zeros_like(targets, dtype=bool)
    candidate = np.zeros_like(targets, dtype=bool)
    fold_rows = []
    for fold, (fit, valid) in enumerate(folds):
        prevalence = targets[fit].mean(0)
        baseline[valid] = _rank_decode(probability[valid], prevalence, 1.10)
        candidate[valid] = _rank_decode(adjusted[valid], prevalence, 1.10)
        old = float(f1_score(targets[valid], baseline[valid], average="macro", zero_division=0))
        new = float(f1_score(targets[valid], candidate[valid], average="macro", zero_division=0))
        fold_rows.append({"fold": fold, "baseline_macro_f1": old,
                          "candidate_macro_f1": new, "delta": new-old})
    old_score = float(f1_score(targets, baseline, average="macro", zero_division=0))
    new_score = float(f1_score(targets, candidate, average="macro", zero_division=0))
    bootstrap = _user_bootstrap(
        targets, baseline, candidate, groups, seed=505050, draws=3000,
    )
    per_label = []
    for label in DEFAULT_BOOSTS:
        old = float(f1_score(targets[:, label], baseline[:, label], zero_division=0))
        new = float(f1_score(targets[:, label], candidate[:, label], zero_division=0))
        per_label.append({
            "label": config.ID2FACTOR[label],
            "support": int(targets[:, label].sum()),
            "anchor_matches": int(flags[:, label].sum()),
            "anchor_true_positives": int((flags[:, label] * targets[:, label]).sum()),
            "anchor_roc_auc": float(roc_auc_score(targets[:, label], flags[:, label])),
            "anchor_pr_auc": float(average_precision_score(targets[:, label], flags[:, label])),
            "baseline_f1": old, "candidate_f1": new, "delta": new-old,
        })
    # The fixed rule bank is definition-derived but development-audited. Require
    # every targeted label to be non-decreasing and at least four outer folds
    # to avoid adopting a gain caused by one rare-label fold.
    adopted = bool(
        new_score >= old_score + .003
        and sum(row["delta"] >= -1e-12 for row in fold_rows) >= 4
        and all(row["delta"] >= -1e-12 for row in per_label)
        and bootstrap["positive_fraction"] >= .75
    )
    payload = {
        "training_version": TRAINING_VERSION,
        "evaluation": "fixed definition boundary policy; five factor-balanced user-disjoint folds",
        "baseline_macro_f1": old_score,
        "candidate_macro_f1": new_score,
        "delta": new_score-old_score,
        "boosts": {config.ID2FACTOR[k]: v for k, v in DEFAULT_BOOSTS.items()},
        "folds": fold_rows, "per_label": per_label,
        "user_cluster_bootstrap": bootstrap,
        "caveat": "Rule bank is definition-derived but development-audited; leaderboard confirmation is required.",
        "adopted": adopted,
    }
    calibration = {
        "training_version": TRAINING_VERSION, "adopted": adopted,
        "boosts": {str(k): float(v) for k, v in DEFAULT_BOOSTS.items()},
        "nested_macro_f1": new_score, "nested_baseline_macro_f1": old_score,
    }
    RESULTS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    CALIBRATION.write_text(json.dumps(calibration, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    return payload


if __name__ == "__main__":
    main()
