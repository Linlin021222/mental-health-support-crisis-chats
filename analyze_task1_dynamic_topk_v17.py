"""OOF-calibrated dynamic evidence count from boundary confidence."""
from __future__ import annotations

import json

import numpy as np
import torch

from analyze_task1_evidence_v4 import _cue_cache, _decoder_grid, _evaluate, _fuse, _search
from baseline import _post_phrase_f1
from configs.config import config
from inference.task1_evidence_v4 import decode_model_evidence, load_evidence_calibration
from utils.task1_metric import task1_score


OUTPUT = config.OUTPUT_DIR / "task1_dynamic_topk_v17"
RESULTS = OUTPUT / "results.json"
CALIBRATION = OUTPUT / "calibration.json"
RULES = ("count", "count_plus_one", "sparse_one_else_three", "sparse_two_else_three")


def _topk(count, rule):
    if rule == "count":
        return max(1, min(3, count))
    if rule == "count_plus_one":
        return max(1, min(3, count + 1))
    if rule == "sparse_one_else_three":
        return 1 if count <= 1 else 3
    if rule == "sparse_two_else_three":
        return 2 if count <= 1 else 3
    raise ValueError(rule)


def _dynamic_scores(records, base_decoded, cues, policy, high_threshold, rule):
    scores = np.empty(len(records), dtype=np.float32); predictions = []
    for index, record in enumerate(records):
        high = decode_model_evidence(
            record["text"], record["offsets"], record["start"], record["end"],
            threshold=high_threshold, max_tokens=12, end_policy="best", limit=5,
        )
        topk = _topk(len(high), rule)
        evidence = _fuse(record, base_decoded[index], cues[index], policy, topk)
        predictions.append(evidence)
        scores[index] = _post_phrase_f1(evidence, record["gold"])
    return scores, predictions


def _dynamic_search(records, decoded_cache, cue_cache):
    rows = []
    # Keep the candidate family deliberately small; calibration has only 325
    # posts and should select count behaviour, not overfit dozens of decoders.
    for decoder_key in ((0.55, 12, "best"), (0.60, 12, "best"), (0.65, 12, "best")):
        for policy in ("predicted_extended_first", "hierarchical_extended_first"):
            for high_threshold in (0.55, 0.60, 0.65, 0.70, 0.75):
                for rule in RULES:
                    scores, _ = _dynamic_scores(
                        records, decoded_cache[decoder_key], cue_cache[policy], policy,
                        high_threshold, rule,
                    )
                    rows.append({
                        "threshold": decoder_key[0], "max_tokens": decoder_key[1],
                        "end_policy": decoder_key[2], "cue_policy": policy,
                        "high_threshold": high_threshold, "count_rule": rule,
                        "phrase_f1": float(scores.mean()),
                    })
    return sorted(rows, key=lambda row: row["phrase_f1"], reverse=True)


def _apply(records, decoded_cache, cue_cache, parameters):
    key = (
        parameters["threshold"], parameters["max_tokens"], parameters["end_policy"]
    )
    return _dynamic_scores(
        records, decoded_cache[key], cue_cache[parameters["cue_policy"]],
        parameters["cue_policy"], parameters["high_threshold"],
        parameters["count_rule"],
    )[0]


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    inner = torch.load(
        config.OUTPUT_DIR / "task1_oof_reranker_v16" / "inner_oof_raw.pt",
        map_location="cpu", weights_only=False,
    )["records"]
    strict = torch.load(
        config.OUTPUT_DIR / "task1_evidence_v4" / "fold0_raw.pt",
        map_location="cpu", weights_only=False,
    )["records"]
    print("v17: decoding inner OOF and outer strict grids...", flush=True)
    inner_decoded = _decoder_grid(inner); inner_cues = _cue_cache(inner)
    strict_decoded = _decoder_grid(strict); strict_cues = _cue_cache(strict)
    selected = _dynamic_search(inner, inner_decoded, inner_cues)[0]
    inner_scores = _apply(inner, inner_decoded, inner_cues, selected)
    strict_scores = _apply(strict, strict_decoded, strict_cues, selected)

    deployed = load_evidence_calibration()
    _, baseline_scores = _evaluate(
        strict, strict_decoded[(
            deployed["threshold"], deployed["max_tokens"], deployed["end_policy"]
        )], strict_cues, np.arange(len(strict)), deployed["cue_policy"], deployed["topk"],
    )
    risk_f1 = float(deployed["strict_risk_f1"])
    local_groups = np.asarray([row["user"] for row in strict]); unique = np.unique(local_groups)
    rng = np.random.default_rng(config.SEED + 1717); deltas = []
    for _ in range(3000):
        sampled_users = rng.choice(unique, size=len(unique), replace=True)
        sampled = np.concatenate([
            np.flatnonzero(local_groups == user) for user in sampled_users
        ])
        deltas.append(float(strict_scores[sampled].mean() - baseline_scores[sampled].mean()))
    bootstrap = {
        "mean_phrase_delta": float(np.mean(deltas)),
        "p05_phrase_delta": float(np.quantile(deltas, 0.05)),
        "p95_phrase_delta": float(np.quantile(deltas, 0.95)),
        "positive_fraction": float(np.mean(np.asarray(deltas) > 0)),
    }
    baseline_phrase = float(baseline_scores.mean()); strict_phrase = float(strict_scores.mean())
    adopted = bool(
        strict_phrase >= baseline_phrase + 0.005
        and bootstrap["positive_fraction"] >= 0.80
    )
    payload = {
        "training_version": "task1-oof-dynamic-topk-v17",
        "selected_on_inner_oof": {**selected, "posts": len(inner)},
        "inner_oof_phrase_f1": float(inner_scores.mean()),
        "baseline": {
            "risk_f1": risk_f1, "phrase_f1": baseline_phrase,
            "task1": task1_score(risk_f1, baseline_phrase),
        },
        "strict": {
            "risk_f1": risk_f1, "phrase_f1": strict_phrase,
            "task1": task1_score(risk_f1, strict_phrase),
            "improved_posts": int((strict_scores > baseline_scores).sum()),
            "worsened_posts": int((strict_scores < baseline_scores).sum()),
        },
        "user_cluster_bootstrap": bootstrap, "adopted": adopted,
    }
    RESULTS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    CALIBRATION.write_text(json.dumps({
        "training_version": payload["training_version"], "adopted": adopted,
        **{key: selected[key] for key in (
            "threshold", "max_tokens", "end_policy", "cue_policy",
            "high_threshold", "count_rule",
        )},
        "strict_phrase_f1": strict_phrase,
        "strict_task1": task1_score(risk_f1, strict_phrase),
        "bootstrap_positive_fraction": bootstrap["positive_fraction"],
    }, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    return payload


if __name__ == "__main__":
    main()
