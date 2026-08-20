"""Diagnostic: use V13 only to prune/reorder the deployed V4 evidence."""
from __future__ import annotations

import json

import numpy as np
import torch
from transformers import AutoTokenizer

from baseline import _post_phrase_f1
from configs.config import config
from inference.task1_evidence_v4 import (
    apply_evidence_policy, decode_model_evidence, load_evidence_calibration,
)
from trainer.task1_evidence_reranker_v13 import (
    CALIBRATION, CHECKPOINT, OUTPUT, PairDataset, _make_model, _normalise,
    _pair_rows, _pool_records, _post_score_map, _score,
)


RESULTS = OUTPUT / "hybrid_diagnostic.json"


def _baseline(record, calibration):
    spans = decode_model_evidence(
        record["text"], record["offsets"], record["start"], record["end"],
        threshold=float(calibration["threshold"]),
        max_tokens=int(calibration["max_tokens"]),
        end_policy=calibration["end_policy"], limit=5,
    )
    return apply_evidence_policy(
        record["text"], record["risk"], spans,
        policy=calibration["cue_policy"], topk=int(calibration["topk"]),
    )


def _phrase_score(phrase, post, scores):
    normal = _normalise(phrase); values = []
    for candidate, score in zip(post["candidates"], scores):
        other = _normalise(candidate["phrase"])
        if normal == other or normal in other or other in normal:
            values.append(float(score))
    return max(values, default=0.5)


def main():
    raw = torch.load(
        config.OUTPUT_DIR / "task1_evidence_v4" / "fold0_raw.pt",
        map_location="cpu", weights_only=False,
    )
    calibration = load_evidence_calibration()
    posts = _pool_records(raw["records"], use_truth=False)
    rows = _pair_rows(posts, np.arange(len(posts)), training=False)
    tokenizer = AutoTokenizer.from_pretrained(
        config.MODEL_NAME, use_fast=True, local_files_only=True
    )
    dataset = PairDataset(rows, tokenizer)
    model = _make_model().to("cuda")
    model.load_state_dict(torch.load(CHECKPOINT, map_location="cuda"))
    pair_scores = _score(model, dataset, torch.device("cuda"), "v13 hybrid scores")
    grouped = _post_score_map(rows, pair_scores)
    baseline = [_baseline(record, calibration) for record in raw["records"]]
    baseline_scores = np.asarray([
        _post_phrase_f1(evidence, record["gold"])
        for evidence, record in zip(baseline, raw["records"])
    ], dtype=np.float32)
    configurations = []
    for mode in ("preserve", "rerank"):
        for topk in (1, 2, 3):
            for threshold in (0.0, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80):
                per_post = []; predictions = []
                for index, (post, evidence) in enumerate(zip(posts, baseline)):
                    scores = [grouped[index].get(i, 0.0) for i in range(len(post["candidates"]))]
                    ranked = [
                        (phrase, _phrase_score(phrase, post, scores)) for phrase in evidence
                    ]
                    if mode == "rerank":
                        ranked.sort(key=lambda item: item[1], reverse=True)
                    selected = [
                        phrase for phrase, score in ranked if score >= threshold
                    ][:topk]
                    predictions.append(selected)
                    per_post.append(_post_phrase_f1(selected, post["gold"]))
                values = np.asarray(per_post, dtype=np.float32)
                configurations.append({
                    "mode": mode, "topk": topk, "threshold": threshold,
                    "phrase_f1": float(values.mean()),
                    "improved_posts": int((values > baseline_scores).sum()),
                    "worsened_posts": int((values < baseline_scores).sum()),
                })
    configurations.sort(key=lambda row: row["phrase_f1"], reverse=True)
    payload = {
        "baseline_phrase_f1": float(baseline_scores.mean()),
        "best": configurations[0], "top20": configurations[:20],
    }
    RESULTS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    return payload


if __name__ == "__main__":
    main()
