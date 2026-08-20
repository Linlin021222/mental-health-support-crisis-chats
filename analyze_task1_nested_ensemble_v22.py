"""Leak-free four-model nested bagging audit for Task 1 evidence."""
from __future__ import annotations

import json

import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

from baseline import _post_phrase_f1
from configs.config import config
from datasets.dataset import SuicideRiskDataset
from inference.task1_evidence_v4 import (
    apply_evidence_policy,
    decode_model_evidence,
    load_evidence_calibration,
)
from models.multitask_model import SuicideRiskMultiTaskModel
from preprocess.preprocess import load_train_data
from trainer.task1_oof_stack_v20 import (
    INNER_FOLDS,
    OUTPUT as V20_OUTPUT,
    _collect_first_stage,
)
from utils.task1_metric import task1_score


OUTPUT = config.OUTPUT_DIR / "task1_nested_ensemble_v22"
RAW = OUTPUT / "outer_predictions.pt"
RESULTS = OUTPUT / "results.json"
CALIBRATION = OUTPUT / "calibration.json"
TRAINING_VERSION = "task1-nested-bagging-v22"
ENSEMBLE_WEIGHT = 0.25


def _collect(dataset, outer_valid, device):
    if RAW.exists():
        saved = torch.load(RAW, map_location="cpu", weights_only=False)
        if (saved.get("training_version") == TRAINING_VERSION
                and np.array_equal(np.asarray(saved["outer_valid"]), outer_valid)):
            print("V22: resumed four-model outer predictions", flush=True)
            return saved["fold_rows"]
    fold_rows = []
    for fold in range(INNER_FOLDS):
        model = SuicideRiskMultiTaskModel().to(device)
        model.backbone.encoder.gradient_checkpointing_disable()
        model.load_state_dict(torch.load(
            V20_OUTPUT / f"inner_fold{fold}_model.pt", map_location=device
        ))
        rows = _collect_first_stage(
            model, dataset, outer_valid, device,
            f"V22 inner model {fold} outer inference",
        )
        fold_rows.append(rows)
        del model
        torch.cuda.empty_cache()
    torch.save({
        "training_version": TRAINING_VERSION,
        "outer_valid": np.asarray(outer_valid), "fold_rows": fold_rows,
    }, RAW)
    return fold_rows


def _evidence(record, start, end, risk, calibration):
    phrases = decode_model_evidence(
        record["text"], record["offsets"], start, end,
        threshold=float(calibration["threshold"]),
        max_tokens=int(calibration["max_tokens"]),
        end_policy=str(calibration["end_policy"]), limit=5,
    )
    return apply_evidence_policy(
        record["text"], int(risk), phrases,
        policy=str(calibration["cue_policy"]),
        topk=int(calibration["topk"]),
    )


def _bootstrap(groups, baseline, candidate):
    unique = np.unique(groups)
    rng = np.random.default_rng(config.SEED + 2222)
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
        raise RuntimeError("V22 requires CUDA")
    dataset = SuicideRiskDataset(config.CACHE_DIR / "train_cache.pt")
    frame = load_train_data().reset_index(drop=True)
    labels = frame.risk_label.to_numpy()
    groups = frame.anon_user_id.astype(str).to_numpy()
    _, outer_valid = next(StratifiedGroupKFold(
        n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED
    ).split(np.zeros(len(frame)), labels, groups))
    stable = torch.load(
        config.OUTPUT_DIR / "task1_evidence_v4" / "fold0_raw.pt",
        map_location="cpu", weights_only=False,
    )
    if not np.array_equal(np.asarray(stable["valid_idx"]), outer_valid):
        raise ValueError("V22 outer rows differ from V4")
    records = stable["records"]
    fold_rows = _collect(dataset, outer_valid, torch.device("cuda"))
    expected_ids = [str(record["row_id"]) for record in records]
    for rows in fold_rows:
        if [str(row["row_id"]) for row in rows] != expected_ids:
            raise ValueError("V22 model predictions are row-misaligned")
    calibration = load_evidence_calibration()
    baseline_scores = []; candidate_scores = []
    mean_start = [];
    mean_end = []
    for index in range(len(records)):
        mean_start.append(torch.stack([
            rows[index]["start"].float() for rows in fold_rows
        ]).mean(0))
        mean_end.append(torch.stack([
            rows[index]["end"].float() for rows in fold_rows
        ]).mean(0))
    for index, record in enumerate(records):
        risk = int(record["risk"])
        baseline = _evidence(
            record, record["start"], record["end"], risk, calibration
        )
        blended_start = (
            (1.0 - ENSEMBLE_WEIGHT) * record["start"].float()
            + ENSEMBLE_WEIGHT * mean_start[index]
        )
        blended_end = (
            (1.0 - ENSEMBLE_WEIGHT) * record["end"].float()
            + ENSEMBLE_WEIGHT * mean_end[index]
        )
        candidate = _evidence(
            record, blended_start, blended_end, risk, calibration
        )
        baseline_scores.append(_post_phrase_f1(baseline, record["gold"]))
        candidate_scores.append(_post_phrase_f1(candidate, record["gold"]))
    baseline_scores = np.asarray(baseline_scores, dtype=np.float32)
    candidate_scores = np.asarray(candidate_scores, dtype=np.float32)
    truth = labels[outer_valid]
    risk = np.asarray([int(record["risk"]) for record in records])
    risk_f1 = float(f1_score(truth, risk, average="weighted", zero_division=0))
    baseline_task1 = task1_score(risk_f1, float(baseline_scores.mean()))
    candidate_task1 = task1_score(risk_f1, float(candidate_scores.mean()))
    bootstrap = _bootstrap(
        groups[outer_valid], baseline_scores, candidate_scores
    )
    adopted = bool(
        candidate_task1 >= baseline_task1 + 0.005
        and bootstrap["positive_fraction"] >= 0.80
    )
    payload = {
        "training_version": TRAINING_VERSION,
        "evaluation_scope": "four inner models and main model all exclude outer users",
        "predeclared_inner_ensemble_weight": ENSEMBLE_WEIGHT,
        "baseline": {
            "risk_f1": risk_f1,
            "phrase_f1": float(baseline_scores.mean()),
            "task1": baseline_task1,
        },
        "candidate": {
            "risk_f1": risk_f1,
            "phrase_f1": float(candidate_scores.mean()),
            "task1": candidate_task1,
            "improved_posts": int((candidate_scores > baseline_scores).sum()),
            "worsened_posts": int((candidate_scores < baseline_scores).sum()),
        },
        "user_cluster_bootstrap": bootstrap,
        "adopted": adopted,
    }
    RESULTS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    CALIBRATION.write_text(json.dumps({
        "training_version": TRAINING_VERSION,
        "adopted": adopted, "ensemble_weight": ENSEMBLE_WEIGHT,
        "strict_task1": candidate_task1,
        "bootstrap_positive_fraction": bootstrap["positive_fraction"],
    }, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    return payload


if __name__ == "__main__":
    main()
