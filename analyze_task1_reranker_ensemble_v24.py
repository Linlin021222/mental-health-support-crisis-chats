"""Nested-OOF ensemble of semantic and position-aware evidence rerankers."""
from __future__ import annotations

import json

import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from transformers import AutoTokenizer

from baseline import _post_phrase_f1
from configs.config import config
from inference.task1_evidence_v4 import load_evidence_calibration
from preprocess.preprocess import load_train_data
from trainer.task1_evidence_reranker_v13 import (
    PairDataset, _make_model, _pair_rows, _pool_records, _post_score_map, _score,
)
from trainer.task1_oof_stack_v20 import (
    INNER_FOLDS, OUTPUT as V20_OUTPUT, RERANKER as V20_MODEL,
    _baseline_evidence, _hybrid_prediction, _parameter_grid,
)
from analyze_task1_position_reranker_v23 import (
    CHECKPOINT as V23_MODEL, _position_pair_rows,
)
from utils.task1_metric import task1_score


OUTPUT = config.OUTPUT_DIR / "task1_reranker_ensemble_v24"
RESULTS = OUTPUT / "results.json"
CALIBRATION = OUTPUT / "calibration.json"
TRAINING_VERSION = "task1-oof-reranker-ensemble-v24"


def _load_inner():
    records = []; membership = {}
    for fold in range(INNER_FOLDS):
        saved = torch.load(V20_OUTPUT / f"inner_fold{fold}_raw.pt",
                           map_location="cpu", weights_only=False)
        for row in saved["records"]:
            records.append(row); membership[int(row["global_index"])] = fold
    records.sort(key=lambda row: row["global_index"])
    return records, np.asarray([
        membership[int(row["global_index"])] for row in records
    ], dtype=np.int64)


def _aligned(first, second):
    return all(
        a["post_index"] == b["post_index"]
        and a["candidate_index"] == b["candidate_index"]
        and a["label"] == b["label"]
        for a, b in zip(first, second)
    ) and len(first) == len(second)


def _model_scores(path, rows, tokenizer, device, description):
    dataset = PairDataset(rows, tokenizer)
    model = _make_model().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    values = _score(model, dataset, device, description)
    del model, dataset; torch.cuda.empty_cache()
    return values


def _bootstrap(groups, baseline, candidate):
    unique = np.unique(groups); rng = np.random.default_rng(config.SEED + 2424)
    values = []
    for _ in range(4000):
        users = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([np.flatnonzero(groups == user) for user in users])
        values.append(float(candidate[indices].mean() - baseline[indices].mean()))
    values = np.asarray(values)
    return {"mean_phrase_delta": float(values.mean()),
            "p05_phrase_delta": float(np.quantile(values, .05)),
            "p95_phrase_delta": float(np.quantile(values, .95)),
            "positive_fraction": float((values > 0).mean())}


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("V24 requires CUDA")
    frame = load_train_data().reset_index(drop=True)
    labels = frame.risk_label.to_numpy(); groups = frame.anon_user_id.astype(str).to_numpy()
    _, outer_valid = next(StratifiedGroupKFold(
        n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED
    ).split(np.zeros(len(frame)), labels, groups))
    inner_records, membership = _load_inner()
    raw = torch.load(config.OUTPUT_DIR / "task1_evidence_v4" / "fold0_raw.pt",
                     map_location="cpu", weights_only=False)
    outer_records = raw["records"]
    calibration = load_evidence_calibration()
    inner_posts = _pool_records(inner_records, use_truth=False)
    outer_posts = _pool_records(outer_records, use_truth=False)
    for post, record in zip(inner_posts, inner_records):
        post["baseline_evidence"] = _baseline_evidence(record, calibration)
    for post, record in zip(outer_posts, outer_records):
        post["baseline_evidence"] = _baseline_evidence(record, calibration)
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME, use_fast=True,
                                               local_files_only=True)
    device = torch.device("cuda")
    calibration_posts = np.flatnonzero(membership == INNER_FOLDS - 1)
    standard_cal = _pair_rows(inner_posts, calibration_posts, training=False)
    position_cal = _position_pair_rows(inner_posts, calibration_posts, training=False)
    if not _aligned(standard_cal, position_cal):
        raise ValueError("V24 calibration candidate rows are misaligned")
    standard_score = _model_scores(V20_MODEL, standard_cal, tokenizer, device,
                                   "V24 semantic calibration")
    position_score = _model_scores(V23_MODEL, position_cal, tokenizer, device,
                                   "V24 position calibration")
    mean_calibration = 0.5 * standard_score + 0.5 * position_score
    selected = _parameter_grid(inner_posts, standard_cal, mean_calibration,
                               calibration_posts, calibration)[0]
    print(f"V24 inner phrase={selected['phrase_f1']:.4f} policy={selected}", flush=True)

    indices = np.arange(len(outer_posts))
    standard_outer = _pair_rows(outer_posts, indices, training=False)
    position_outer = _position_pair_rows(outer_posts, indices, training=False)
    if not _aligned(standard_outer, position_outer):
        raise ValueError("V24 outer candidate rows are misaligned")
    semantic = _model_scores(V20_MODEL, standard_outer, tokenizer, device,
                             "V24 semantic outer")
    position = _model_scores(V23_MODEL, position_outer, tokenizer, device,
                             "V24 position outer")
    scores = 0.5 * semantic + 0.5 * position
    grouped = _post_score_map(standard_outer, scores)
    baseline = []; candidate = []
    for index, post in enumerate(outer_posts):
        old = list(post["baseline_evidence"])
        values = [grouped[index].get(j, 0.0) for j in range(len(post["candidates"]))]
        new = _hybrid_prediction(post, values, old, selected)
        baseline.append(_post_phrase_f1(old, post["gold"]))
        candidate.append(_post_phrase_f1(new, post["gold"]))
    baseline = np.asarray(baseline, dtype=np.float32)
    candidate = np.asarray(candidate, dtype=np.float32)
    truth = labels[outer_valid]; risk = np.asarray([int(row["risk"]) for row in outer_records])
    risk_f1 = float(f1_score(truth, risk, average="weighted", zero_division=0))
    baseline_task1 = task1_score(risk_f1, float(baseline.mean()))
    candidate_task1 = task1_score(risk_f1, float(candidate.mean()))
    bootstrap = _bootstrap(groups[outer_valid], baseline, candidate)
    adopted = bool(candidate_task1 >= baseline_task1 + .005
                   and bootstrap["positive_fraction"] >= .80)
    payload = {"training_version": TRAINING_VERSION,
               "evaluation_scope": "equal-weight ensemble selected on inner OOF only",
               "selected": selected,
               "baseline": {"risk_f1": risk_f1, "phrase_f1": float(baseline.mean()),
                            "task1": baseline_task1},
               "candidate": {"risk_f1": risk_f1, "phrase_f1": float(candidate.mean()),
                             "task1": candidate_task1,
                             "improved_posts": int((candidate > baseline).sum()),
                             "worsened_posts": int((candidate < baseline).sum())},
               "user_cluster_bootstrap": bootstrap, "adopted": adopted}
    RESULTS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    CALIBRATION.write_text(json.dumps({"training_version": TRAINING_VERSION,
        "adopted": adopted, "strict_task1": candidate_task1,
        "bootstrap_positive_fraction": bootstrap["positive_fraction"],
        **{key: selected[key] for key in ("mode", "topk", "threshold", "gate")}},
        indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    return payload


if __name__ == "__main__":
    main()
