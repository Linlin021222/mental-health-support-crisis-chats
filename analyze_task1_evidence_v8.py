"""Nested user-disjoint evaluation of source-aware evidence fusion V8."""
from __future__ import annotations

import json
from collections import Counter

import numpy as np
import torch
from sklearn.model_selection import GroupKFold
from tqdm import tqdm

from analyze_task1_evidence_v4 import _cue_cache, _decoder_grid
from baseline import _post_phrase_f1
from configs.config import config
from inference.task1_evidence_v4 import load_evidence_calibration
from inference.task1_evidence_v8 import smart_fuse_evidence
from utils.task1_metric import task1_score


OUTPUT = config.OUTPUT_DIR / "task1_evidence_v8"
RAW = config.OUTPUT_DIR / "task1_evidence_v4" / "fold0_raw.pt"
RESULTS = OUTPUT / "results.json"
CALIBRATION = OUTPUT / "calibration.json"
TRAINING_VERSION = "task1-source-aware-evidence-v8"
MODES = (
    "model_only", "model_context", "model_strong_only",
    "anchor_model_context", "anchor_model_strong", "strong_anchor_model",
)


def _candidate_scores(records, decoded_cache, cue_cache):
    parameters, columns = [], []
    # Both cue sets are label-conditioned.  The hierarchical version adds
    # lower-severity language and is retained as a bounded ablation.
    cue_policies = ("predicted_extended_first", "hierarchical_extended_first")
    total = len(decoded_cache) * len(cue_policies) * len(MODES) * 4
    with tqdm(total=total, desc="evidence-v8 candidates", unit="cfg") as bar:
        for (threshold, max_tokens, end_policy), decoded in decoded_cache.items():
            for cue_policy in cue_policies:
                cues = cue_cache[cue_policy]
                for mode in MODES:
                    for topk in (1, 2, 3, 4):
                        scores = np.empty(len(records), dtype=np.float32)
                        for index, row in enumerate(records):
                            evidence = smart_fuse_evidence(
                                row["text"], row["risk"], decoded[index], cues[index],
                                mode=mode, topk=topk,
                            )
                            scores[index] = _post_phrase_f1(evidence, row["gold"])
                        parameters.append({
                            "threshold": float(threshold),
                            "max_tokens": int(max_tokens),
                            "end_policy": end_policy,
                            "cue_policy": cue_policy,
                            "fusion_mode": mode,
                            "topk": int(topk),
                        })
                        columns.append(scores)
                        bar.update()
    return parameters, np.stack(columns)


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
        "fusion_mode": _mode([row["fusion_mode"] for row in rows]),
        "topk": int(_median_member([row["topk"] for row in rows])),
    }


def _parameter_key(row):
    return (
        float(row["threshold"]), int(row["max_tokens"]), row["end_policy"],
        row["cue_policy"], row["fusion_mode"], int(row["topk"]),
    )


def _bootstrap(groups, baseline, candidate, rounds=3000):
    unique = np.unique(groups)
    rng = np.random.default_rng(config.SEED + 808)
    deltas = np.empty(rounds, dtype=np.float32)
    for iteration in range(rounds):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([np.flatnonzero(groups == user) for user in sampled])
        deltas[iteration] = candidate[indices].mean() - baseline[indices].mean()
    return {
        "mean_phrase_delta": float(deltas.mean()),
        "p05_phrase_delta": float(np.quantile(deltas, 0.05)),
        "p95_phrase_delta": float(np.quantile(deltas, 0.95)),
        "positive_fraction": float(np.mean(deltas > 0)),
    }


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    print("evidence-v8: loading strict cached logits...", flush=True)
    raw = torch.load(RAW, map_location="cpu", weights_only=False)
    records = raw["records"]
    groups = np.asarray([row["user"] for row in records])
    indices = np.arange(len(records))
    calibration = load_evidence_calibration()
    if calibration is None:
        raise FileNotFoundError("Task 1 evidence-v4 calibration is missing or not adopted")

    print("evidence-v8: decoding span grid...", flush=True)
    decoded_cache = _decoder_grid(records)
    cue_cache = _cue_cache(records)
    parameters, scores = _candidate_scores(records, decoded_cache, cue_cache)
    lookup = {_parameter_key(row): i for i, row in enumerate(parameters)}

    # Compare to the exact deployed V4 predictions, not to a reconstructed
    # approximation from the new candidate family.
    from analyze_task1_evidence_v4 import _evaluate
    base_decoded = decoded_cache[(
        float(calibration["threshold"]), int(calibration["max_tokens"]),
        calibration["end_policy"],
    )]
    _, baseline_scores = _evaluate(
        records, base_decoded, cue_cache, indices,
        calibration["cue_policy"], int(calibration["topk"]),
    )

    crossfit = np.empty(len(records), dtype=np.float32)
    selections = []
    splitter = GroupKFold(n_splits=4)
    for fold, (fit, held) in enumerate(splitter.split(indices, groups=groups)):
        mean_scores = scores[:, fit].mean(axis=1)
        best_index = int(np.argmax(mean_scores))
        crossfit[held] = scores[best_index, held]
        selections.append({
            "fold": fold, **parameters[best_index],
            "fit_phrase_f1": float(mean_scores[best_index]),
            "heldout_phrase_f1": float(scores[best_index, held].mean()),
            "heldout_posts": int(len(held)),
        })
        print(
            f"evidence-v8: fold {fold + 1}/4 "
            f"heldout_phrase_f1={selections[-1]['heldout_phrase_f1']:.4f}",
            flush=True,
        )

    fixed = _aggregate(selections)
    fixed_index = lookup[_parameter_key(fixed)]
    fixed_scores = scores[fixed_index]
    optimistic_index = int(np.argmax(scores.mean(axis=1)))
    risk_f1 = float(calibration["strict_risk_f1"])
    baseline_phrase = float(baseline_scores.mean())
    crossfit_phrase = float(crossfit.mean())
    fixed_phrase = float(fixed_scores.mean())
    bootstrap = _bootstrap(groups, baseline_scores, fixed_scores)

    # Adoption demands nested improvement and convincing user-cluster support.
    adopted = bool(
        crossfit_phrase >= baseline_phrase + 0.005
        and fixed_phrase >= baseline_phrase + 0.005
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
            "task1": task1_score(risk_f1, crossfit_phrase), "folds": selections,
        },
        "fixed_production": {
            "parameters": fixed, "phrase_f1": fixed_phrase,
            "task1": task1_score(risk_f1, fixed_phrase),
        },
        "optimistic_full_holdout": {
            **parameters[optimistic_index],
            "phrase_f1": float(scores[optimistic_index].mean()),
            "task1": task1_score(risk_f1, float(scores[optimistic_index].mean())),
        },
        "diagnostics": {
            "fixed_improved_posts": int((fixed_scores > baseline_scores).sum()),
            "fixed_worsened_posts": int((fixed_scores < baseline_scores).sum()),
            "fixed_unchanged_posts": int((fixed_scores == baseline_scores).sum()),
        },
        "user_cluster_bootstrap": bootstrap,
        "adopted": adopted,
    }
    RESULTS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    CALIBRATION.write_text(json.dumps({
        "training_version": TRAINING_VERSION, "adopted": adopted,
        **fixed, "strict_risk_f1": risk_f1,
        "strict_baseline_phrase_f1": baseline_phrase,
        "strict_crossfit_phrase_f1": crossfit_phrase,
        "strict_fixed_phrase_f1": fixed_phrase,
        "strict_fixed_task1": task1_score(risk_f1, fixed_phrase),
        "bootstrap_positive_fraction": bootstrap["positive_fraction"],
    }, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    return payload


if __name__ == "__main__":
    main()
