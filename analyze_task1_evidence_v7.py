"""Label-conditional evidence calibration on the strict user holdout.

Ideation, Behavior, and Attempt evidence have different lengths and cue
density.  V4 used one global decoder; V7 selects one decoder per predicted
risk label inside nested user folds, with shrinkage toward global performance.
"""
from __future__ import annotations

import json
from collections import Counter

import numpy as np
import torch
from sklearn.model_selection import GroupKFold
from tqdm import tqdm

from analyze_task1_evidence_v4 import POLICIES, _cue_cache, _decoder_grid, _fuse
from baseline import _post_phrase_f1
from configs.config import config
from inference.task1_evidence_v4 import load_evidence_calibration
from utils.task1_metric import task1_score


OUTPUT = config.OUTPUT_DIR / "task1_evidence_v7"
RAW = config.OUTPUT_DIR / "task1_evidence_v4" / "fold0_raw.pt"
RESULTS = OUTPUT / "results.json"
CALIBRATION = OUTPUT / "calibration.json"
TRAINING_VERSION = "task1-evidence-label-conditional-v7"
SHRINKAGE_POSTS = 8.0


def _candidate_scores(records, decoded_cache, cue_cache):
    parameters, columns = [], []
    total = len(decoded_cache) * len(POLICIES) * 5
    with tqdm(total=total, desc="evidence-v7 candidates", unit="cfg") as progress:
        for (threshold, max_tokens, end_policy), decoded in decoded_cache.items():
            for policy in POLICIES:
                cues = cue_cache[policy]
                for topk in (1, 2, 3, 4, 5):
                    scores = np.empty(len(records), dtype=np.float32)
                    for index, row in enumerate(records):
                        evidence = _fuse(row, decoded[index], cues[index], policy, topk)
                        scores[index] = _post_phrase_f1(evidence, row["gold"])
                    parameters.append({
                        "threshold": float(threshold), "max_tokens": int(max_tokens),
                        "end_policy": end_policy, "cue_policy": policy,
                        "topk": int(topk),
                    })
                    columns.append(scores); progress.update()
    return parameters, np.stack(columns)


def _parameter_key(row):
    return (
        float(row["threshold"]), int(row["max_tokens"]), row["end_policy"],
        row["cue_policy"], int(row["topk"]),
    )


def _mode(values):
    return Counter(values).most_common(1)[0][0]


def _median_member(values):
    median = float(np.median(values))
    return min(values, key=lambda value: (abs(float(value) - median), float(value)))


def _aggregate(rows):
    return {
        "threshold": float(_median_member([row["threshold"] for row in rows])),
        "max_tokens": int(_median_member([row["max_tokens"] for row in rows])),
        "end_policy": _mode([row["end_policy"] for row in rows]),
        "cue_policy": _mode([row["cue_policy"] for row in rows]),
        "topk": int(_median_member([row["topk"] for row in rows])),
    }


def _select(scores, fit_label, fit_all):
    """Empirical-Bayes shrinkage prevents tiny Attempt subsets overfitting."""
    label_sum = scores[:, fit_label].sum(1)
    global_mean = scores[:, fit_all].mean(1)
    objective = (
        label_sum + SHRINKAGE_POSTS * global_mean
    ) / (len(fit_label) + SHRINKAGE_POSTS)
    return int(np.argmax(objective))


def _bootstrap(groups, baseline, candidate, rounds=2000):
    unique = np.unique(groups); rng = np.random.default_rng(config.SEED + 707)
    deltas = []
    for _ in range(rounds):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([np.flatnonzero(groups == user) for user in sampled])
        deltas.append(float(candidate[indices].mean() - baseline[indices].mean()))
    values = np.asarray(deltas)
    return {
        "mean_phrase_delta": float(values.mean()),
        "p05_phrase_delta": float(np.quantile(values, 0.05)),
        "p95_phrase_delta": float(np.quantile(values, 0.95)),
        "positive_fraction": float(np.mean(values > 0)),
    }


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    print("evidence-v7: loading saved strict predictions...", flush=True)
    raw = torch.load(RAW, map_location="cpu", weights_only=False)
    records = raw["records"]
    groups = np.asarray([row["user"] for row in records])
    predicted_risk = np.asarray([int(row["risk"]) for row in records])
    indices = np.arange(len(records))
    calibration = load_evidence_calibration()
    if calibration is None:
        raise FileNotFoundError("Task 1 evidence-v4 calibration is not adopted")

    print("evidence-v7: decoding model spans...", flush=True)
    decoded_cache = _decoder_grid(records)
    cue_cache = _cue_cache(records)
    parameters, scores = _candidate_scores(records, decoded_cache, cue_cache)
    lookup = {_parameter_key(row): index for index, row in enumerate(parameters)}
    baseline_parameters = {
        key: calibration[key]
        for key in ("threshold", "max_tokens", "end_policy", "cue_policy", "topk")
    }
    baseline_index = lookup[_parameter_key(baseline_parameters)]
    baseline_scores = scores[baseline_index]

    crossfit = np.zeros(len(records), dtype=np.float32)
    selections = {risk_id: [] for risk_id in range(config.NUM_RISK_CLASSES)}
    folds = []
    splitter = GroupKFold(n_splits=4)
    for fold, (fit, held) in enumerate(splitter.split(indices, groups=groups)):
        print(f"evidence-v7: user fold {fold + 1}/4...", flush=True)
        fold_rows = []
        for risk_id in range(config.NUM_RISK_CLASSES):
            fit_label = fit[predicted_risk[fit] == risk_id]
            held_label = held[predicted_risk[held] == risk_id]
            if risk_id == config.RISK_LABELS["Indicator"] or len(fit_label) == 0:
                selected_index = baseline_index
            else:
                selected_index = _select(scores, fit_label, fit)
            selected = parameters[selected_index]
            selections[risk_id].append(selected)
            crossfit[held_label] = scores[selected_index, held_label]
            fold_rows.append({
                "risk": config.ID2RISK[risk_id],
                "fit_posts": int(len(fit_label)), "heldout_posts": int(len(held_label)),
                **selected,
                "heldout_phrase_f1": (
                    float(scores[selected_index, held_label].mean())
                    if len(held_label) else None
                ),
            })
        folds.append({"fold": fold, "labels": fold_rows})

    fixed_parameters, fixed_scores = {}, np.zeros(len(records), dtype=np.float32)
    optimistic = {}
    for risk_id in range(config.NUM_RISK_CLASSES):
        label_indices = indices[predicted_risk == risk_id]
        if risk_id == config.RISK_LABELS["Indicator"]:
            fixed = baseline_parameters
        else:
            fixed = _aggregate(selections[risk_id])
        fixed_index = lookup[_parameter_key(fixed)]
        fixed_scores[label_indices] = scores[fixed_index, label_indices]
        fixed_parameters[config.ID2RISK[risk_id]] = fixed
        best_index = (
            int(np.argmax(scores[:, label_indices].mean(1)))
            if len(label_indices) else baseline_index
        )
        optimistic[config.ID2RISK[risk_id]] = {
            **parameters[best_index],
            "posts": int(len(label_indices)),
            "phrase_f1": (
                float(scores[best_index, label_indices].mean())
                if len(label_indices) else None
            ),
        }

    risk_f1 = float(calibration["strict_risk_f1"])
    baseline_phrase = float(baseline_scores.mean())
    crossfit_phrase = float(crossfit.mean())
    fixed_phrase = float(fixed_scores.mean())
    bootstrap = _bootstrap(groups, baseline_scores, fixed_scores)
    adopted = bool(
        crossfit_phrase >= baseline_phrase + 0.010
        and fixed_phrase >= baseline_phrase + 0.010
        and bootstrap["positive_fraction"] >= 0.80
    )
    payload = {
        "training_version": TRAINING_VERSION,
        "baseline": {
            "risk_f1": risk_f1, "phrase_f1": baseline_phrase,
            "task1": task1_score(risk_f1, baseline_phrase),
        },
        "nested_crossfit": {
            "phrase_f1": crossfit_phrase,
            "task1": task1_score(risk_f1, crossfit_phrase), "folds": folds,
        },
        "fixed_production": {
            "parameters_by_predicted_risk": fixed_parameters,
            "phrase_f1": fixed_phrase,
            "task1": task1_score(risk_f1, fixed_phrase),
        },
        "optimistic_per_label": optimistic,
        "user_cluster_bootstrap": bootstrap,
        "adopted": adopted,
    }
    RESULTS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    CALIBRATION.write_text(json.dumps({
        "training_version": TRAINING_VERSION, "adopted": adopted,
        "parameters_by_predicted_risk": fixed_parameters,
        "strict_baseline_phrase_f1": baseline_phrase,
        "strict_crossfit_phrase_f1": crossfit_phrase,
        "strict_fixed_phrase_f1": fixed_phrase,
        "strict_fixed_task1": task1_score(risk_f1, fixed_phrase),
        "bootstrap_positive_fraction": bootstrap["positive_fraction"],
    }, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return payload


if __name__ == "__main__":
    main()
