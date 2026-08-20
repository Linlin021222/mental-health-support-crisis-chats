"""Nested user-level gate for the paper-inspired V58 trajectory expert."""
from __future__ import annotations

import json

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold

from baseline import _post_phrase_f1
from configs.config import config
from preprocess.preprocess import load_train_data
from trainer.task1_factor_trajectory_v58 import (
    PREDICTIONS, _current_baseline, _outer_split, _risk,
)
from trainer.task1_local_counterfactual_train_v56 import _decode
from utils.task1_metric import task1_score


OUTPUT = config.OUTPUT_DIR / "task1_factor_trajectory_gate_v59"
RESULTS = OUTPUT / "results.json"


def _policies():
    rows = []
    for weight in (0.0, .03, .05, .07, .10):
        for preserve_attempt in ((False, True) if weight else (False,)):
            for adjacent_only in ((False, True) if weight else (False,)):
                rows.append({"weight": weight,
                             "preserve_attempt": preserve_attempt,
                             "adjacent_only": adjacent_only})
    return rows


def _apply(texts, baseline_probability, trajectory_probability, policy):
    raw = _risk(
        texts,
        (1. - policy["weight"]) * baseline_probability
        + policy["weight"] * trajectory_probability,
    )
    baseline = _risk(texts, baseline_probability)
    result = raw.copy()
    if policy["preserve_attempt"]:
        result[baseline == config.RISK_LABELS["Attempt"]] = config.RISK_LABELS["Attempt"]
    if policy["adjacent_only"]:
        large = np.abs(result - baseline) > 1
        result[large] = baseline[large]
    return result


def _score(truth, risk, phrase, positions):
    risk_f1 = float(f1_score(
        truth[positions], risk[positions], average="weighted", zero_division=0,
    ))
    phrase_f1 = float(phrase[positions].mean())
    return {"risk_f1": risk_f1, "phrase_f1": phrase_f1,
            "task1": task1_score(risk_f1, phrase_f1)}


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if not PREDICTIONS.exists():
        raise FileNotFoundError("Run --mode task1-factor-trajectory-v58 first")
    frame = load_train_data().reset_index(drop=True)
    labels = frame.risk_label.to_numpy(dtype=np.int64)
    groups = frame.anon_user_id.astype(str).to_numpy()
    train_idx, valid_idx = _outer_split(frame)
    saved = np.load(PREDICTIONS)
    if not np.array_equal(saved["valid_idx"], valid_idx):
        raise RuntimeError("V58 prediction fold mismatch")
    baseline_probability = saved["baseline_probability"]
    trajectory_probability = saved["trajectory_probability"]
    texts = frame.text.iloc[valid_idx].astype(str).tolist()
    truth = labels[valid_idx]
    local_groups = groups[valid_idx]

    # Recreate the accepted V57 evidence logits once. Each candidate policy
    # changes only the risk-conditional decoder, not the evidence model.
    _, records, starts, ends, parameters = _current_baseline(frame, train_idx, valid_idx)
    gold = [list(frame.iloc[int(index)].evidence) for index in valid_idx]
    candidates = []
    for policy in _policies():
        risk = _apply(texts, baseline_probability, trajectory_probability, policy)
        evidence = _decode(records, risk, starts, ends, parameters)
        phrase = np.asarray([
            _post_phrase_f1(predicted, target)
            for predicted, target in zip(evidence, gold)
        ], dtype=np.float32)
        candidates.append({"policy": policy, "risk": risk, "phrase": phrase})

    splits = list(StratifiedGroupKFold(
        n_splits=4, shuffle=True, random_state=config.SEED + 5959,
    ).split(np.zeros(len(valid_idx)), truth, local_groups))
    crossfit_risk = np.zeros(len(valid_idx), dtype=np.int64)
    crossfit_phrase = np.zeros(len(valid_idx), dtype=np.float32)
    selections = []
    for fold, (fit_pos, test_pos) in enumerate(splits):
        ranked = []
        for candidate in candidates:
            metric = _score(truth, candidate["risk"], candidate["phrase"], fit_pos)
            changes = int(np.sum(candidate["risk"][fit_pos]
                                 != candidates[0]["risk"][fit_pos]))
            ranked.append({"candidate": candidate, "metric": metric,
                           "changes": changes})
        # Prefer the simplest policy when the calibration users cannot
        # distinguish two candidates materially.
        selected = max(ranked, key=lambda row: (
            round(row["metric"]["task1"], 4), -row["changes"],
            -row["candidate"]["policy"]["weight"],
        ))
        candidate = selected["candidate"]
        crossfit_risk[test_pos] = candidate["risk"][test_pos]
        crossfit_phrase[test_pos] = candidate["phrase"][test_pos]
        selections.append({
            "fold": fold, "fit_task1": selected["metric"]["task1"],
            "fit_changes": selected["changes"], **candidate["policy"],
            "heldout_posts": int(len(test_pos)),
        })

    base_risk = candidates[0]["risk"]; base_phrase = candidates[0]["phrase"]
    positions = np.arange(len(valid_idx))
    baseline = _score(truth, base_risk, base_phrase, positions)
    candidate = _score(truth, crossfit_risk, crossfit_phrase, positions)

    unique = np.unique(local_groups); rng = np.random.default_rng(595959)
    deltas = []
    for _ in range(4000):
        sampled = rng.choice(unique, len(unique), replace=True)
        sample = np.concatenate([
            np.flatnonzero(local_groups == user) for user in sampled
        ])
        old_f1 = f1_score(truth[sample], base_risk[sample],
                          average="weighted", zero_division=0)
        new_f1 = f1_score(truth[sample], crossfit_risk[sample],
                          average="weighted", zero_division=0)
        deltas.append(task1_score(new_f1, crossfit_phrase[sample].mean())
                      - task1_score(old_f1, base_phrase[sample].mean()))
    deltas = np.asarray(deltas)
    bootstrap = {"mean_delta": float(deltas.mean()),
                 "p05_delta": float(np.quantile(deltas, .05)),
                 "p95_delta": float(np.quantile(deltas, .95)),
                 "positive_fraction": float((deltas > 0).mean())}
    adopted = bool(candidate["task1"] >= baseline["task1"] + .003
                   and bootstrap["positive_fraction"] >= .80
                   and bootstrap["p05_delta"] >= 0.)
    payload = {
        "training_version": "task1-nested-factor-trajectory-gate-v59",
        "evaluation_scope": "nested four-fold user cross-fit inside untouched outer users",
        "baseline": baseline,
        "candidate": {**candidate,
                      "changed_risk": int(np.sum(crossfit_risk != base_risk)),
                      "confusion": confusion_matrix(
                          truth, crossfit_risk, labels=np.arange(4)).tolist()},
        "fold_selections": selections,
        "user_cluster_bootstrap": bootstrap,
        "adopted": adopted,
    }
    RESULTS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    return payload


if __name__ == "__main__":
    main()
