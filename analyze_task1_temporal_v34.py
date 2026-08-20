"""Leak-free user-trajectory decoding for Task 1 risk (V34)."""
from __future__ import annotations

import json

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

from baseline import _post_phrase_f1
from configs.config import config
from inference.task1_evidence_v4 import correct_risk_only, load_evidence_calibration
from preprocess.preprocess import load_train_data
from trainer.task1_atomic_v25 import _bootstrap, _load_records
from trainer.task1_oof_stack_v20 import _baseline_evidence
from utils.task1_metric import task1_score


OUTPUT = config.OUTPUT_DIR / "task1_temporal_v34"
RESULTS = OUTPUT / "results.json"
CALIBRATION = OUTPUT / "calibration.json"
TRAINING_VERSION = "task1-user-trajectory-v34"


def _ordered(frame, indices):
    blocks = []
    local = {int(index): position for position, index in enumerate(indices)}
    subset = frame.iloc[indices]
    for _, rows in subset.groupby("anon_user_id", sort=False):
        ordered = sorted((int(index) for index in rows.index),
                         key=lambda index: (float(frame.iloc[index].post_id), index))
        blocks.append(np.asarray([local[index] for index in ordered], dtype=np.int64))
    return blocks


def _transition(frame, fit_indices, smoothing=1.0):
    counts = np.full((4, 4), smoothing, dtype=np.float64)
    prior = np.full(4, smoothing, dtype=np.float64)
    for _, rows in frame.iloc[fit_indices].groupby("anon_user_id", sort=False):
        rows = rows.sort_values("post_id")
        labels = rows.risk_label.to_numpy(dtype=np.int64)
        if len(labels):
            prior[labels[0]] += 1
        for first, second in zip(labels[:-1], labels[1:]):
            counts[first, second] += 1
    return np.log(counts / counts.sum(1, keepdims=True)), np.log(prior / prior.sum())


def _viterbi(probability, blocks, transition, prior, weight):
    prediction = np.argmax(probability, axis=1)
    for positions in blocks:
        emission = np.log(np.clip(probability[positions], 1e-8, 1.0))
        score = emission[0] + weight * prior; back = []
        for step in range(1, len(positions)):
            values = score[:, None] + weight * transition
            parents = values.argmax(0); score = emission[step] + values[parents, np.arange(4)]
            back.append(parents)
        state = int(score.argmax()); path = [state]
        for parents in reversed(back):
            state = int(parents[state]); path.append(state)
        prediction[positions] = list(reversed(path))
    return prediction


def _local(probability, blocks, weight, radius):
    blended = probability.copy()
    for positions in blocks:
        for offset, position in enumerate(positions):
            lo = max(0, offset - radius); hi = min(len(positions), offset + radius + 1)
            neighbours = np.delete(positions[lo:hi], offset - lo)
            if len(neighbours):
                blended[position] = ((1. - weight) * probability[position]
                                     + weight * probability[neighbours].mean(0))
    return blended.argmax(1)


def _predict(frame, indices, records, parameters, fit_indices):
    probability = np.vstack([records[int(index)]["old_probability"] for index in indices])
    blocks = _ordered(frame, indices)
    if parameters["mode"] == "baseline":
        raw = np.asarray([int(records[int(index)]["risk"]) for index in indices])
    elif parameters["mode"] == "viterbi":
        transition, prior = _transition(frame, fit_indices)
        raw = _viterbi(probability, blocks, transition, prior, parameters["weight"])
    else:
        raw = _local(probability, blocks, parameters["weight"], parameters["radius"])
    return np.asarray([correct_risk_only(records[int(index)]["text"], int(risk))
                       for index, risk in zip(indices, raw)], dtype=np.int64)


def _evidence_scores(frame, indices, records, risks, calibration):
    values = []
    for index, risk in zip(indices, risks):
        record = dict(records[int(index)]); record["risk"] = int(risk)
        evidence = _baseline_evidence(record, calibration)
        values.append(_post_phrase_f1(evidence, list(frame.iloc[int(index)].evidence)))
    return np.asarray(values, dtype=np.float32)


def _grid(frame, fit_idx, valid_idx, records, labels, calibration):
    parameters = [{"mode": "baseline", "weight": 0., "radius": 0}]
    parameters += [{"mode": "viterbi", "weight": weight, "radius": 0}
                   for weight in (.025, .05, .10, .20, .35, .50, .75, 1.)]
    parameters += [{"mode": "local", "weight": weight, "radius": radius}
                   for radius in (1, 2, 3) for weight in (.05, .10, .20, .30, .40)]
    rows = []
    for item in parameters:
        risk = _predict(frame, valid_idx, records, item, fit_idx)
        phrase = _evidence_scores(frame, valid_idx, records, risk, calibration)
        risk_f1 = float(f1_score(labels[valid_idx], risk, average="weighted", zero_division=0))
        rows.append({**item, "risk_f1": risk_f1, "phrase_f1": float(phrase.mean()),
                     "task1": task1_score(risk_f1, float(phrase.mean()))})
    return sorted(rows, key=lambda row: row["task1"], reverse=True)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame = load_train_data().reset_index(drop=True)
    labels = frame.risk_label.to_numpy(); groups = frame.anon_user_id.astype(str).to_numpy()
    outer_train, _ = next(StratifiedGroupKFold(
        n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED
    ).split(np.zeros(len(frame)), labels, groups))
    raw_records, membership, _ = _load_records()
    records = {int(row["global_index"]): row for row in raw_records}
    calibration = load_evidence_calibration()
    cal_idx = np.asarray([i for i in outer_train if membership[int(i)] == 3])
    fit_idx = np.asarray([i for i in outer_train if membership[int(i)] != 3])
    rows = _grid(frame, fit_idx, cal_idx, records, labels, calibration)
    selected = {key: rows[0][key] for key in ("mode", "weight", "radius")}
    print(f"V34 selected on fold3: {rows[0]}", flush=True)

    all_indices = []; all_old_risk = []; all_new_risk = []
    all_old_phrase = []; all_new_phrase = []; folds = []
    for fold in (0, 1, 2):
        valid_idx = np.asarray([i for i in outer_train if membership[int(i)] == fold])
        fit_idx = np.asarray([i for i in outer_train if membership[int(i)] != fold])
        baseline_parameters = {"mode": "baseline", "weight": 0., "radius": 0}
        old_risk = _predict(frame, valid_idx, records, baseline_parameters, fit_idx)
        new_risk = _predict(frame, valid_idx, records, selected, fit_idx)
        old_phrase = _evidence_scores(frame, valid_idx, records, old_risk, calibration)
        new_phrase = _evidence_scores(frame, valid_idx, records, new_risk, calibration)
        old_f1 = float(f1_score(labels[valid_idx], old_risk, average="weighted", zero_division=0))
        new_f1 = float(f1_score(labels[valid_idx], new_risk, average="weighted", zero_division=0))
        folds.append({"fold": fold, "posts": int(len(valid_idx)),
                      "baseline_risk_f1": old_f1, "candidate_risk_f1": new_f1,
                      "risk_delta": new_f1 - old_f1,
                      "baseline_phrase_f1": float(old_phrase.mean()),
                      "candidate_phrase_f1": float(new_phrase.mean()),
                      "changed_posts": int((new_risk != old_risk).sum())})
        print(f"V34 fold={fold} risk {old_f1:.6f} -> {new_f1:.6f}", flush=True)
        all_indices.extend(map(int, valid_idx)); all_old_risk.extend(old_risk)
        all_new_risk.extend(new_risk); all_old_phrase.extend(old_phrase); all_new_phrase.extend(new_phrase)

    order = np.argsort(all_indices); indices = np.asarray(all_indices)[order]
    old_risk = np.asarray(all_old_risk, dtype=np.int64)[order]
    new_risk = np.asarray(all_new_risk, dtype=np.int64)[order]
    old_phrase = np.asarray(all_old_phrase, dtype=np.float32)[order]
    new_phrase = np.asarray(all_new_phrase, dtype=np.float32)[order]
    old_f1 = float(f1_score(labels[indices], old_risk, average="weighted", zero_division=0))
    new_f1 = float(f1_score(labels[indices], new_risk, average="weighted", zero_division=0))
    old_task = task1_score(old_f1, float(old_phrase.mean()))
    new_task = task1_score(new_f1, float(new_phrase.mean()))
    # Bootstrap full Task1 deltas because risk changes as well as evidence.
    unique = np.unique(groups[indices]); rng = np.random.default_rng(config.SEED + 3434); deltas = []
    for _ in range(4000):
        sampled_users = rng.choice(unique, size=len(unique), replace=True)
        positions = np.concatenate([np.flatnonzero(groups[indices] == user) for user in sampled_users])
        old_sample_f1 = f1_score(labels[indices][positions], old_risk[positions],
                                 average="weighted", zero_division=0)
        new_sample_f1 = f1_score(labels[indices][positions], new_risk[positions],
                                 average="weighted", zero_division=0)
        deltas.append(task1_score(new_sample_f1, float(new_phrase[positions].mean()))
                      - task1_score(old_sample_f1, float(old_phrase[positions].mean())))
    deltas = np.asarray(deltas)
    bootstrap = {"mean_task1_delta": float(deltas.mean()),
                 "p05_task1_delta": float(np.quantile(deltas, .05)),
                 "p95_task1_delta": float(np.quantile(deltas, .95)),
                 "positive_fraction": float((deltas > 0).mean())}
    adopted = bool(new_task >= old_task + .003 and bootstrap["positive_fraction"] >= .80)
    payload = {"training_version": TRAINING_VERSION,
        "evaluation_scope": "trajectory policy selected fold3; untouched user folds0-2",
        "selected": selected, "calibration_top10": rows[:10], "folds": folds,
        "baseline": {"risk_f1": old_f1, "phrase_f1": float(old_phrase.mean()), "task1": old_task},
        "candidate": {"risk_f1": new_f1, "phrase_f1": float(new_phrase.mean()), "task1": new_task,
                      "changed_posts": int((new_risk != old_risk).sum())},
        "user_cluster_bootstrap": bootstrap, "adopted": adopted}
    RESULTS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    CALIBRATION.write_text(json.dumps({"training_version": TRAINING_VERSION,
        "adopted": adopted, "selected": selected, "crossvalidated_task1": new_task},
        indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True); return payload


if __name__ == "__main__":
    main()
