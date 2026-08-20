"""Nested calibration of conservative weights for the trained V14 seed."""
from __future__ import annotations

import json

import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

from configs.config import config
from datasets.dataset import SuicideRiskDataset
from inference.task1_evidence_v4 import load_evidence_calibration
from trainer.task1_seed_ensemble_v14 import (
    OUTPUT, PREDICTIONS, _evidence_scores, _risk_predictions, _stable_components,
)
from utils.task1_metric import task1_score


RESULTS = OUTPUT / "weighted_results.json"
CALIBRATION = OUTPUT / "weighted_calibration.json"
WEIGHTS = (0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40)


def main():
    dataset = SuicideRiskDataset(config.CACHE_DIR / "train_cache.pt")
    labels = np.asarray([int(row["risk_label"]) for row in dataset.data])
    groups = np.asarray([str(row["anon_user_id"]) for row in dataset.data])
    train_idx, valid_idx = next(StratifiedGroupKFold(
        n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED
    ).split(np.zeros(len(dataset)), labels, groups))
    raw = torch.load(
        config.OUTPUT_DIR / "task1_evidence_v4" / "fold0_raw.pt",
        map_location="cpu", weights_only=False,
    )
    seed2 = torch.load(PREDICTIONS, map_location="cpu", weights_only=False)["rows"]
    old, v2, lexical = _stable_components(
        dataset, train_idx, valid_idx, raw["records"], torch.device("cuda")
    )
    new = np.vstack([row["probability"] for row in seed2])
    calibration = load_evidence_calibration()
    risks = {}
    for weight in WEIGHTS:
        seed_mix = (1.0 - weight) * old + weight * new
        probability = 0.7 * (0.8 * seed_mix + 0.2 * v2) + 0.3 * lexical
        risks[weight] = _risk_predictions(raw["records"], probability)
    phrase = {}
    for risk_weight in WEIGHTS:
        for evidence_weight in WEIGHTS:
            starts = [
                (1.0 - evidence_weight) * row["start"] + evidence_weight * other["start"]
                for row, other in zip(raw["records"], seed2)
            ]
            ends = [
                (1.0 - evidence_weight) * row["end"] + evidence_weight * other["end"]
                for row, other in zip(raw["records"], seed2)
            ]
            phrase[(risk_weight, evidence_weight)] = _evidence_scores(
                raw["records"], risks[risk_weight], starts, ends, calibration
            )
    truth = labels[valid_idx]; indices = np.arange(len(valid_idx))
    local_groups = groups[valid_idx]

    def metric(subset, risk_weight, evidence_weight):
        prediction = risks[risk_weight]
        risk_f1 = float(f1_score(
            truth[subset], prediction[subset], average="weighted", zero_division=0
        ))
        phrase_f1 = float(phrase[(risk_weight, evidence_weight)][subset].mean())
        return risk_f1, phrase_f1, task1_score(risk_f1, phrase_f1)

    crossfit_risk = np.empty(len(indices), dtype=np.int64)
    crossfit_phrase = np.empty(len(indices), dtype=np.float32); folds = []
    for fold, (fit, held) in enumerate(GroupKFold(n_splits=4).split(
        indices, groups=local_groups
    )):
        candidates = []
        for risk_weight in WEIGHTS:
            for evidence_weight in WEIGHTS:
                candidates.append((
                    metric(fit, risk_weight, evidence_weight)[2],
                    risk_weight, evidence_weight,
                ))
        _, risk_weight, evidence_weight = max(candidates)
        crossfit_risk[held] = risks[risk_weight][held]
        crossfit_phrase[held] = phrase[(risk_weight, evidence_weight)][held]
        held_metric = metric(held, risk_weight, evidence_weight)
        folds.append({
            "fold": fold, "risk_seed_weight": risk_weight,
            "evidence_seed_weight": evidence_weight,
            "heldout_risk_f1": held_metric[0],
            "heldout_phrase_f1": held_metric[1], "heldout_task1": held_metric[2],
        })
        print(f"v14 weighted fold {fold + 1}/4 task1={held_metric[2]:.4f}", flush=True)
    def median_member(values):
        median = float(np.median(values))
        return float(min(WEIGHTS, key=lambda value: (abs(value - median), value)))
    risk_weight = median_member([row["risk_seed_weight"] for row in folds])
    evidence_weight = median_member([row["evidence_seed_weight"] for row in folds])
    fixed = metric(indices, risk_weight, evidence_weight)
    baseline = metric(indices, 0.0, 0.0)
    crossfit_risk_f1 = float(f1_score(
        truth, crossfit_risk, average="weighted", zero_division=0
    ))
    crossfit_phrase_f1 = float(crossfit_phrase.mean())
    crossfit_task1 = task1_score(crossfit_risk_f1, crossfit_phrase_f1)
    optimistic = max(
        ({
            "risk_seed_weight": rw, "evidence_seed_weight": ew,
            "risk_f1": metric(indices, rw, ew)[0],
            "phrase_f1": metric(indices, rw, ew)[1],
            "task1": metric(indices, rw, ew)[2],
        } for rw in WEIGHTS for ew in WEIGHTS),
        key=lambda row: row["task1"],
    )
    rng = np.random.default_rng(config.SEED + 1415); unique = np.unique(local_groups)
    deltas = []
    fixed_prediction = risks[risk_weight]; fixed_phrase = phrase[(risk_weight, evidence_weight)]
    baseline_prediction = risks[0.0]; baseline_phrase = phrase[(0.0, 0.0)]
    for _ in range(3000):
        sampled_users = rng.choice(unique, size=len(unique), replace=True)
        sampled = np.concatenate([
            np.flatnonzero(local_groups == user) for user in sampled_users
        ])
        old_risk = f1_score(
            truth[sampled], baseline_prediction[sampled], average="weighted", zero_division=0
        )
        new_risk_score = f1_score(
            truth[sampled], fixed_prediction[sampled], average="weighted", zero_division=0
        )
        deltas.append(
            task1_score(new_risk_score, float(fixed_phrase[sampled].mean()))
            - task1_score(old_risk, float(baseline_phrase[sampled].mean()))
        )
    bootstrap = {
        "mean_delta": float(np.mean(deltas)),
        "p05_delta": float(np.quantile(deltas, 0.05)),
        "p95_delta": float(np.quantile(deltas, 0.95)),
        "positive_fraction": float(np.mean(np.asarray(deltas) > 0)),
    }
    adopted = bool(
        crossfit_task1 >= baseline[2] + 0.003
        and fixed[2] >= baseline[2] + 0.003
        and bootstrap["positive_fraction"] >= 0.75
    )
    payload = {
        "training_version": "task1-two-seed-weighted-v14b",
        "baseline": {"risk_f1": baseline[0], "phrase_f1": baseline[1], "task1": baseline[2]},
        "nested_crossfit": {
            "risk_f1": crossfit_risk_f1, "phrase_f1": crossfit_phrase_f1,
            "task1": crossfit_task1, "folds": folds,
        },
        "fixed_production": {
            "risk_seed_weight": risk_weight, "evidence_seed_weight": evidence_weight,
            "risk_f1": fixed[0], "phrase_f1": fixed[1], "task1": fixed[2],
        },
        "optimistic_full_holdout": optimistic,
        "user_cluster_bootstrap": bootstrap, "adopted": adopted,
    }
    RESULTS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    CALIBRATION.write_text(json.dumps({
        "training_version": payload["training_version"], "adopted": adopted,
        "risk_seed_weight": risk_weight, "evidence_seed_weight": evidence_weight,
        "strict_fixed_task1": fixed[2],
        "bootstrap_positive_fraction": bootstrap["positive_fraction"],
    }, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    return payload


if __name__ == "__main__":
    main()
