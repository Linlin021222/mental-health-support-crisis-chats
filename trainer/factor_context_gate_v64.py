"""Nested, label-specific same-user context gate for Task 2.

All posts from a leaderboard user are available together, but train and test
users never overlap.  This experiment therefore uses only *other posts of the
same evaluation user*.  Context is selected independently per factor in four
inner user-disjoint folds and evaluated in five outer user-disjoint folds.
The accepted base probability is always retained when a label is unstable.
"""
from __future__ import annotations

import json
from collections import Counter

import numpy as np
from sklearn.metrics import f1_score

from configs.config import config
from preprocess.preprocess import load_train_data
from trainer.factor_balanced_calibration_v47 import _components
from trainer.factor_sentence_evidence_cv_v27 import _user_bootstrap
from utils.multilabel_group_split import multilabel_group_folds, split_audit


OUTPUT = config.OUTPUT_DIR / "factor_context_gate_v64"
RESULTS = OUTPUT / "results.json"
CALIBRATION = OUTPUT / "calibration.json"
TRAINING_VERSION = "label-specific-user-context-v64"
METHODS = ("mean_other", "top2_other", "max_other")
CONTEXT_WEIGHTS = (0.05, 0.10, 0.15, 0.20)
RATIOS = (0.90, 1.00, 1.10, 1.20, 1.30)
BASE_RATIO = 1.10


def user_aggregate(probability, groups, method):
    probability = np.asarray(probability, dtype=np.float32)
    groups = np.asarray(groups).astype(str)
    result = probability.copy()
    for group in np.unique(groups):
        rows = np.flatnonzero(groups == group)
        if len(rows) < 2:
            continue
        values = probability[rows]
        if method == "mean_other":
            result[rows] = (values.sum(0, keepdims=True) - values) / (len(rows) - 1)
        else:
            for local, row in enumerate(rows):
                other = np.delete(values, local, axis=0)
                if method == "max_other":
                    result[row] = other.max(0)
                elif method == "top2_other":
                    result[row] = np.sort(other, axis=0)[-min(2, len(other)):].mean(0)
                else:
                    raise ValueError(method)
    return result


def _rank(values):
    values = np.asarray(values, dtype=np.float32)
    order = np.argsort(values, kind="stable")
    result = np.empty(len(values), dtype=np.float32)
    result[order] = np.arange(len(values), dtype=np.float32) / max(1, len(values)-1)
    return result


def _topk(values, prevalence, ratio):
    count = max(1, min(len(values), int(round(len(values) * prevalence * ratio))))
    selected = np.argpartition(values, len(values)-count)[len(values)-count:]
    result = np.zeros(len(values), dtype=bool); result[selected] = True
    return result


def _crossfit_one(score, aggregates, targets, groups, risk, label, seed):
    inner = multilabel_group_folds(targets, groups, risk, 4, seed)
    candidates = [("none", 0.0, BASE_RATIO)] + [
        (method, weight, ratio) for method in METHODS
        for weight in CONTEXT_WEIGHTS for ratio in RATIOS
    ]
    rows = []
    for method, weight, ratio in candidates:
        prediction = np.zeros(len(targets), dtype=bool); fold_values = []
        for fit, valid in inner:
            own = _rank(score[valid])
            if method == "none":
                mixed = own
            else:
                context = _rank(aggregates[method][valid, label])
                mixed = (1.0-weight) * own + weight * context
            prediction[valid] = _topk(
                mixed, float(targets[fit, label].mean()), ratio,
            )
            fold_values.append(float(f1_score(
                targets[valid, label], prediction[valid], zero_division=0,
            )))
        value = float(f1_score(targets[:, label], prediction, zero_division=0))
        rows.append({
            "method": method, "weight": float(weight), "ratio": float(ratio),
            "f1": value, "fold_f1": fold_values,
            "objective": value - .010*weight - .002*abs(ratio-BASE_RATIO),
        })
    baseline = rows[0]
    selected = max(rows, key=lambda row: row["objective"])
    nonworse = sum(a >= b - 1e-12 for a, b in zip(
        selected["fold_f1"], baseline["fold_f1"],
    ))
    accepted = bool(
        selected["method"] != "none"
        and selected["f1"] >= baseline["f1"] + .010
        and nonworse >= 3
    )
    if not accepted:
        selected = dict(baseline)
    selected.update({
        "accepted": accepted, "nonworse_inner_folds": int(nonworse),
        "baseline_inner_f1": float(baseline["f1"]),
    })
    return selected


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame = load_train_data().reset_index(drop=True)
    components, targets = _components()
    probability = components["current"].astype(np.float32)
    groups = frame.anon_user_id.astype(str).to_numpy()
    risk = frame.risk_label.to_numpy(dtype=np.int64)
    folds = multilabel_group_folds(
        targets, groups, risk, config.N_FOLDS, config.SEED + 64,
    )
    baseline = np.zeros_like(targets, dtype=bool)
    candidate = np.zeros_like(targets, dtype=bool)
    fold_rows = []
    for fold, (fit, valid) in enumerate(folds):
        fit_aggregates = {
            method: user_aggregate(probability[fit], groups[fit], method)
            for method in METHODS
        }
        valid_aggregates = {
            method: user_aggregate(probability[valid], groups[valid], method)
            for method in METHODS
        }
        selections = []
        for label in range(config.NUM_FACTORS):
            selected = _crossfit_one(
                probability[fit, label], fit_aggregates, targets[fit],
                groups[fit], risk[fit], label,
                config.SEED + 6400 + 31*fold + label,
            )
            prevalence = float(targets[fit, label].mean())
            own = _rank(probability[valid, label])
            baseline[valid, label] = _topk(own, prevalence, BASE_RATIO)
            if selected["method"] == "none":
                mixed = own
            else:
                context = _rank(valid_aggregates[selected["method"]][:, label])
                mixed = ((1.0-selected["weight"])*own
                         + selected["weight"]*context)
            candidate[valid, label] = _topk(mixed, prevalence, selected["ratio"])
            selections.append({
                **selected, "label": config.ID2FACTOR[label],
                "support": int(targets[fit, label].sum()),
            })
        old = float(f1_score(targets[valid], baseline[valid], average="macro", zero_division=0))
        new = float(f1_score(targets[valid], candidate[valid], average="macro", zero_division=0))
        fold_rows.append({
            "fold": fold, "baseline_macro_f1": old,
            "candidate_macro_f1": new, "delta": new-old,
            "selections": selections,
        })
        print(f"V64 fold={fold}: {old:.6f} -> {new:.6f} ({new-old:+.6f})", flush=True)
    old_score = float(f1_score(targets, baseline, average="macro", zero_division=0))
    new_score = float(f1_score(targets, candidate, average="macro", zero_division=0))
    bootstrap = _user_bootstrap(
        targets, baseline, candidate, groups, seed=646464, draws=4000,
    )
    production = []
    for label in range(config.NUM_FACTORS):
        selections = [row["selections"][label] for row in fold_rows]
        accepted = [row for row in selections if row["accepted"]]
        methods = Counter(row["method"] for row in accepted)
        method, votes = methods.most_common(1)[0] if methods else ("none", 0)
        matching = [row for row in accepted if row["method"] == method]
        stable = bool(votes >= 4)
        production.append({
            "label": config.ID2FACTOR[label],
            "method": method if stable else "none",
            "weight": float(np.median([x["weight"] for x in matching])) if stable else 0.0,
            "ratio": float(np.median([x["ratio"] for x in matching])) if stable else BASE_RATIO,
            "outer_votes": int(votes), "accepted": stable,
        })
    adopted = bool(
        new_score >= old_score + .003
        and bootstrap["positive_fraction"] >= .80
        and bootstrap["p05_delta"] >= 0
        and any(row["accepted"] for row in production)
    )
    per_label = []
    for label in range(config.NUM_FACTORS):
        old = float(f1_score(targets[:, label], baseline[:, label], zero_division=0))
        new = float(f1_score(targets[:, label], candidate[:, label], zero_division=0))
        per_label.append({
            "label": config.ID2FACTOR[label], "support": int(targets[:, label].sum()),
            "baseline_f1": old, "candidate_f1": new, "delta": new-old,
        })
    payload = {
        "training_version": TRAINING_VERSION,
        "evaluation": "five outer / four inner factor-balanced user-disjoint folds",
        "split_audit": split_audit(folds, targets, groups),
        "baseline_macro_f1": old_score, "candidate_macro_f1": new_score,
        "delta": new_score-old_score, "user_cluster_bootstrap": bootstrap,
        "folds": fold_rows, "per_label": per_label,
        "production_parameters": production, "adopted": adopted,
    }
    RESULTS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    CALIBRATION.write_text(json.dumps({
        "training_version": TRAINING_VERSION, "adopted": adopted,
        "baseline_macro_f1": old_score, "candidate_macro_f1": new_score,
        "parameters": production, "bootstrap": bootstrap,
    }, indent=2), encoding="utf-8")
    print(json.dumps({
        "training_version": TRAINING_VERSION,
        "baseline_macro_f1": old_score, "candidate_macro_f1": new_score,
        "delta": new_score-old_score, "bootstrap": bootstrap,
        "adopted": adopted,
    }, indent=2), flush=True)
    return payload


if __name__ == "__main__":
    main()
