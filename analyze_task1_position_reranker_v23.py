"""Position-aware nested-OOF evidence reranker inspired by CLPsych 2024."""
from __future__ import annotations

import json
import re

import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from transformers import AutoTokenizer

from baseline import _post_phrase_f1
from configs.config import config
from datasets.dataset import SuicideRiskDataset
from inference.task1_evidence_v4 import load_evidence_calibration
from preprocess.preprocess import load_train_data
from trainer.task1_evidence_reranker_v13 import (
    PairDataset,
    _context,
    _pool_records,
    _post_score_map,
    _score,
)
from trainer.task1_oof_stack_v20 import (
    INNER_FOLDS,
    OUTPUT as V20_OUTPUT,
    _baseline_evidence,
    _hybrid_prediction,
    _parameter_grid,
    _train_reranker,
)
from utils.task1_metric import task1_score


OUTPUT = config.OUTPUT_DIR / "task1_position_reranker_v23"
CHECKPOINT = OUTPUT / "reranker.pt"
RESULTS = OUTPUT / "results.json"
CALIBRATION = OUTPUT / "calibration.json"
TRAINING_VERSION = "task1-position-aware-oof-v23"


def _position_features(text, phrase):
    match = re.search(re.escape(str(phrase)), str(text), flags=re.IGNORECASE)
    start = match.start() if match else len(str(text)) // 2
    ratio = start / max(1, len(str(text)))
    if start < 120 or ratio < 0.08:
        bucket = "opening"
    elif ratio < 0.30:
        bucket = "early"
    elif ratio < 0.70:
        bucket = "middle"
    else:
        bucket = "late"
    words = len(str(phrase).split())
    length = "short" if words <= 4 else "medium" if words <= 10 else "long"
    return bucket, ratio, length


def _position_pair_rows(posts, post_indices, training=False):
    rows = []
    for post_index in post_indices:
        post = posts[int(post_index)]
        positives = []; negatives = []
        for candidate_index, candidate in enumerate(post["candidates"]):
            phrase = candidate["phrase"]
            label = float(_post_phrase_f1([phrase], post["gold"]) > 0)
            position, ratio, length = _position_features(post["text"], phrase)
            votes = int(candidate["votes"])
            agreement = "high" if votes >= 15 else "medium" if votes >= 5 else "low"
            prompt = (
                f"Predicted risk: {config.ID2RISK[int(post['risk'])]}. "
                f"Candidate evidence: {phrase}. "
                f"Document position: {position}. Relative position: {ratio:.2f}. "
                f"Span length: {length}. Decoder agreement: {agreement}."
            )
            item = {
                "post_index": int(post_index),
                "candidate_index": candidate_index,
                "prompt": prompt,
                "context": _context(post["text"], phrase),
                "label": label,
            }
            (positives if label else negatives).append(item)
        if training:
            negatives = negatives[:max(5, min(10, 3 * max(1, len(positives))))]
        rows.extend(positives + negatives)
    return rows


def _load_inner():
    records = []; membership = {}
    for fold in range(INNER_FOLDS):
        saved = torch.load(
            V20_OUTPUT / f"inner_fold{fold}_raw.pt",
            map_location="cpu", weights_only=False,
        )
        for row in saved["records"]:
            records.append(row); membership[int(row["global_index"])] = fold
    records.sort(key=lambda row: row["global_index"])
    folds = np.asarray([
        membership[int(row["global_index"])] for row in records
    ], dtype=np.int64)
    return records, folds


def _bootstrap(groups, baseline, candidate):
    unique = np.unique(groups); rng = np.random.default_rng(config.SEED + 2323)
    values = []
    for _ in range(4000):
        sampled_users = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([
            np.flatnonzero(groups == user) for user in sampled_users
        ])
        values.append(float(candidate[indices].mean() - baseline[indices].mean()))
    values = np.asarray(values)
    return {
        "mean_phrase_delta": float(values.mean()),
        "p05_phrase_delta": float(np.quantile(values, 0.05)),
        "p95_phrase_delta": float(np.quantile(values, 0.95)),
        "positive_fraction": float((values > 0).mean()),
    }


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("V23 requires CUDA")
    frame = load_train_data().reset_index(drop=True)
    dataset = SuicideRiskDataset(config.CACHE_DIR / "train_cache.pt")
    labels = frame.risk_label.to_numpy()
    groups = frame.anon_user_id.astype(str).to_numpy()
    _, outer_valid = next(StratifiedGroupKFold(
        n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED
    ).split(np.zeros(len(frame)), labels, groups))
    inner_records, membership = _load_inner()
    raw = torch.load(
        config.OUTPUT_DIR / "task1_evidence_v4" / "fold0_raw.pt",
        map_location="cpu", weights_only=False,
    )
    if not np.array_equal(np.asarray(raw["valid_idx"]), outer_valid):
        raise ValueError("V23 outer rows differ from V4")
    outer_records = raw["records"]
    evidence_calibration = load_evidence_calibration()
    inner_posts = _pool_records(inner_records, use_truth=False)
    outer_posts = _pool_records(outer_records, use_truth=False)
    for post, record in zip(inner_posts, inner_records):
        post["baseline_evidence"] = _baseline_evidence(record, evidence_calibration)
    for post, record in zip(outer_posts, outer_records):
        post["baseline_evidence"] = _baseline_evidence(record, evidence_calibration)
    tokenizer = AutoTokenizer.from_pretrained(
        config.MODEL_NAME, use_fast=True, local_files_only=True
    )
    model, selected, history, fit_pairs, calibration_pairs = _train_reranker(
        inner_posts, membership, tokenizer, torch.device("cuda"),
        evidence_calibration, row_builder=_position_pair_rows,
        checkpoint_path=CHECKPOINT,
    )
    outer_rows = _position_pair_rows(
        outer_posts, np.arange(len(outer_posts)), training=False
    )
    outer_dataset = PairDataset(outer_rows, tokenizer)
    pair_scores = _score(
        model, outer_dataset, torch.device("cuda"), "V23 outer strict"
    )
    grouped = _post_score_map(outer_rows, pair_scores)
    baseline = []; candidate = []
    for index, post in enumerate(outer_posts):
        old = list(post["baseline_evidence"])
        values = [
            grouped[index].get(candidate_index, 0.0)
            for candidate_index in range(len(post["candidates"]))
        ]
        new = _hybrid_prediction(post, values, old, selected)
        baseline.append(_post_phrase_f1(old, post["gold"]))
        candidate.append(_post_phrase_f1(new, post["gold"]))
    baseline = np.asarray(baseline, dtype=np.float32)
    candidate = np.asarray(candidate, dtype=np.float32)
    truth = labels[outer_valid]
    risk = np.asarray([int(record["risk"]) for record in outer_records])
    risk_f1 = float(f1_score(truth, risk, average="weighted", zero_division=0))
    baseline_task1 = task1_score(risk_f1, float(baseline.mean()))
    candidate_task1 = task1_score(risk_f1, float(candidate.mean()))
    bootstrap = _bootstrap(groups[outer_valid], baseline, candidate)
    adopted = bool(
        candidate_task1 >= baseline_task1 + 0.005
        and bootstrap["positive_fraction"] >= 0.80
    )
    payload = {
        "training_version": TRAINING_VERSION,
        "evaluation_scope": "position features selected on nested OOF; untouched outer users",
        "gold_position_audit": {
            "within_first_100_chars": 0.4341,
            "within_first_200_chars": 0.6108,
        },
        "fit_pairs": fit_pairs, "calibration_pairs": calibration_pairs,
        "history": history, "selected": selected,
        "baseline": {
            "risk_f1": risk_f1, "phrase_f1": float(baseline.mean()),
            "task1": baseline_task1,
        },
        "candidate": {
            "risk_f1": risk_f1, "phrase_f1": float(candidate.mean()),
            "task1": candidate_task1,
            "improved_posts": int((candidate > baseline).sum()),
            "worsened_posts": int((candidate < baseline).sum()),
        },
        "user_cluster_bootstrap": bootstrap,
        "adopted": adopted,
    }
    RESULTS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    CALIBRATION.write_text(json.dumps({
        "training_version": TRAINING_VERSION, "adopted": adopted,
        "strict_task1": candidate_task1,
        "bootstrap_positive_fraction": bootstrap["positive_fraction"],
        **{key: selected[key] for key in ("mode", "topk", "threshold", "gate")},
    }, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    return payload


if __name__ == "__main__":
    main()
