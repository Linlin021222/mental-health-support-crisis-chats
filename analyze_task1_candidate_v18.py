"""Consolidated V18 development candidate and user-cluster uncertainty audit."""
from __future__ import annotations

import json

import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

from analyze_task1_lexical_v11 import (
    _lexical_experts, _prediction, _transformer_probability,
)
from baseline import _post_phrase_f1
from configs.config import config
from datasets.dataset import SuicideRiskDataset
from inference.task1_evidence_v4 import (
    apply_evidence_policy, decode_model_evidence, load_evidence_calibration,
)
from preprocess.preprocess import load_train_data
from utils.task1_metric import task1_score


OUTPUT = config.OUTPUT_DIR / "task1_candidate_v18"
RESULTS = OUTPUT / "results.json"
CALIBRATION = OUTPUT / "calibration.json"
SEED2_WEIGHT = 0.20


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
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
    seed2 = torch.load(
        config.OUTPUT_DIR / "task1_seed_ensemble_v14" / "seed2_valid.pt",
        map_location="cpu", weights_only=False,
    )["rows"]
    frame = load_train_data().reset_index(drop=True)
    transformer = _transformer_probability(dataset, valid_idx, raw["records"])
    experts = _lexical_experts(frame, train_idx, valid_idx)
    risk_parameters = {
        "expert": "svc-c0.25-balanced", "temperature": 1.0,
        "lexical_weight": 0.60, "attempt_bias": 0.20,
    }
    prediction = _prediction(
        raw["records"], transformer,
        experts[risk_parameters["expert"]], risk_parameters,
    )
    v7 = json.loads((
        config.OUTPUT_DIR / "task1_evidence_v7" / "results.json"
    ).read_text(encoding="utf-8"))
    evidence_parameters = v7["optimistic_per_label"]
    candidate_scores = np.empty(len(raw["records"]), dtype=np.float32)
    for index, (record, second, risk) in enumerate(zip(raw["records"], seed2, prediction)):
        parameters = evidence_parameters[config.ID2RISK[int(risk)]]
        start = (1.0 - SEED2_WEIGHT) * record["start"] + SEED2_WEIGHT * second["start"]
        end = (1.0 - SEED2_WEIGHT) * record["end"] + SEED2_WEIGHT * second["end"]
        spans = decode_model_evidence(
            record["text"], record["offsets"], start, end,
            threshold=parameters["threshold"],
            max_tokens=parameters["max_tokens"],
            end_policy=parameters["end_policy"], limit=5,
        )
        evidence = apply_evidence_policy(
            record["text"], int(risk), spans,
            policy=parameters["cue_policy"], topk=parameters["topk"],
        )
        candidate_scores[index] = _post_phrase_f1(evidence, record["gold"])

    deployed = load_evidence_calibration()
    baseline_prediction = np.asarray([int(row["risk"]) for row in raw["records"]])
    baseline_scores = np.empty(len(raw["records"]), dtype=np.float32)
    for index, record in enumerate(raw["records"]):
        spans = decode_model_evidence(
            record["text"], record["offsets"], record["start"], record["end"],
            threshold=deployed["threshold"], max_tokens=deployed["max_tokens"],
            end_policy=deployed["end_policy"], limit=5,
        )
        evidence = apply_evidence_policy(
            record["text"], baseline_prediction[index], spans,
            policy=deployed["cue_policy"], topk=deployed["topk"],
        )
        baseline_scores[index] = _post_phrase_f1(evidence, record["gold"])
    truth = labels[valid_idx]
    baseline_risk = float(f1_score(
        truth, baseline_prediction, average="weighted", zero_division=0
    ))
    candidate_risk = float(f1_score(
        truth, prediction, average="weighted", zero_division=0
    ))
    baseline_task1 = task1_score(baseline_risk, float(baseline_scores.mean()))
    candidate_task1 = task1_score(candidate_risk, float(candidate_scores.mean()))

    local_groups = groups[valid_idx]; unique = np.unique(local_groups)
    rng = np.random.default_rng(config.SEED + 1818); deltas = []
    for _ in range(5000):
        sampled_users = rng.choice(unique, size=len(unique), replace=True)
        sampled = np.concatenate([
            np.flatnonzero(local_groups == user) for user in sampled_users
        ])
        old_risk = f1_score(
            truth[sampled], baseline_prediction[sampled], average="weighted", zero_division=0
        )
        new_risk = f1_score(
            truth[sampled], prediction[sampled], average="weighted", zero_division=0
        )
        deltas.append(
            task1_score(new_risk, float(candidate_scores[sampled].mean()))
            - task1_score(old_risk, float(baseline_scores[sampled].mean()))
        )
    bootstrap = {
        "mean_delta": float(np.mean(deltas)),
        "p05_delta": float(np.quantile(deltas, 0.05)),
        "p95_delta": float(np.quantile(deltas, 0.95)),
        "positive_fraction": float(np.mean(np.asarray(deltas) > 0)),
    }
    # This is an adopted development candidate, not a fresh unbiased estimate:
    # its component parameters were selected during repeated strict-fold work.
    adopted_for_full_training = bool(
        candidate_task1 >= 0.80 and bootstrap["positive_fraction"] >= 0.80
    )
    payload = {
        "training_version": "task1-consolidated-candidate-v18",
        "evaluation_scope": "user-disjoint development holdout (parameters previously compared)",
        "baseline": {
            "risk_f1": baseline_risk, "phrase_f1": float(baseline_scores.mean()),
            "task1": baseline_task1,
        },
        "candidate": {
            "risk_f1": candidate_risk, "phrase_f1": float(candidate_scores.mean()),
            "task1": candidate_task1,
            "improved_phrase_posts": int((candidate_scores > baseline_scores).sum()),
            "worsened_phrase_posts": int((candidate_scores < baseline_scores).sum()),
        },
        "risk_parameters": risk_parameters,
        "evidence_parameters_by_predicted_risk": evidence_parameters,
        "seed2_evidence_weight": SEED2_WEIGHT,
        "user_cluster_bootstrap": bootstrap,
        "adopted_for_full_training": adopted_for_full_training,
        "unbiased_test_claim": False,
    }
    RESULTS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    CALIBRATION.write_text(json.dumps({
        "training_version": payload["training_version"],
        "adopted": adopted_for_full_training,
        "unbiased_test_claim": False,
        **risk_parameters,
        "seed2_evidence_weight": SEED2_WEIGHT,
        "evidence_parameters_by_predicted_risk": evidence_parameters,
        "strict_development_task1": candidate_task1,
        "bootstrap_positive_fraction": bootstrap["positive_fraction"],
    }, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    return payload


if __name__ == "__main__":
    main()
