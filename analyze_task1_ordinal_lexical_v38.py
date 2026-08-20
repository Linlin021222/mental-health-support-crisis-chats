"""OOF ordinal lexical expert for Task 1 risk (V38)."""
from __future__ import annotations

import json
import re

import joblib
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.svm import LinearSVC

from analyze_task1_lexical_v11 import _softmax
from analyze_task1_oof_risk_v36 import (
    CACHE as V36_CACHE, _evidence_matrix, _predict as _v36_predict,
)
from baseline import _vectorizer
from configs.config import config
from inference.task1_evidence_v4 import correct_risk_only
from preprocess.preprocess import load_train_data
from trainer.task1_atomic_v25 import _load_records
from utils.task1_metric import task1_score


OUTPUT = config.OUTPUT_DIR / "task1_ordinal_lexical_v38"
RESULTS = OUTPUT / "results.json"
CALIBRATION = OUTPUT / "calibration.json"
CACHE = OUTPUT / "oof_decisions.npz"
ARTIFACT = OUTPUT / "full_ordinal_lexical.joblib"
TRAINING_VERSION = "task1-oof-ordinal-lexical-v38"
C_VALUES = (.25, .5, 1., 2.)


def _train_decisions(frame, outer_train, records, membership):
    global_indices = np.asarray([int(row["global_index"]) for row in records])
    if CACHE.exists():
        saved = np.load(CACHE)
        if np.array_equal(saved["global_indices"], global_indices):
            print("V38 resumed ordinal lexical OOF decisions", flush=True)
            return saved["decisions"]
    local = {index: position for position, index in enumerate(global_indices)}
    result = np.zeros((len(C_VALUES), len(records), 3), dtype=np.float32)
    for fold in range(4):
        fit = np.asarray([i for i in outer_train if membership[int(i)] != fold])
        valid = np.asarray([i for i in outer_train if membership[int(i)] == fold])
        vectorizer = _vectorizer(); train = vectorizer.fit_transform(frame.text.iloc[fit])
        held = vectorizer.transform(frame.text.iloc[valid]); labels = frame.risk_label.to_numpy()[fit]
        positions = [local[int(i)] for i in valid]
        for c_index, c_value in enumerate(C_VALUES):
            for threshold in range(3):
                target = (labels > threshold).astype(np.int64)
                model = LinearSVC(C=c_value, class_weight="balanced").fit(train, target)
                result[c_index, positions, threshold] = model.decision_function(held)
        print(f"V38 ordinal OOF fold {fold + 1}/4", flush=True)
    np.savez_compressed(CACHE, global_indices=global_indices, decisions=result)
    return result


def _ordinal_probability(decision, temperature):
    cumulative = 1. / (1. + np.exp(-np.asarray(decision) / float(temperature)))
    # Enforce P(y>0) >= P(y>1) >= P(y>2) for every post.
    cumulative = np.sort(cumulative, axis=1)[:, ::-1]
    probability = np.column_stack([
        1. - cumulative[:, 0], cumulative[:, 0] - cumulative[:, 1],
        cumulative[:, 1] - cumulative[:, 2], cumulative[:, 2],
    ]).clip(0.)
    return probability / probability.sum(1, keepdims=True).clip(1e-8)


def _predict(texts, transformer, nominal_decision, ordinal_decision, parameters):
    nominal = _softmax(nominal_decision, 1.)
    base = .3 * transformer + .7 * nominal
    ordinal = _ordinal_probability(ordinal_decision, parameters["temperature"])
    probability = ((1. - parameters["ordinal_weight"]) * base
                   + parameters["ordinal_weight"] * ordinal)
    logits = np.log(np.clip(probability, 1e-8, 1.))
    logits[:, 0] -= .15; logits[:, 2] += .20; logits[:, 3] += .40
    raw = logits.argmax(1)
    return np.asarray([correct_risk_only(text, int(risk))
                       for text, risk in zip(texts, raw)], dtype=np.int64)


def _metric(truth, prediction, evidence, indices):
    indices = np.asarray(indices, dtype=np.int64)
    risk = float(f1_score(truth[indices], prediction[indices],
                          average="weighted", zero_division=0))
    phrase_values = evidence[np.arange(len(prediction)), prediction]
    phrase = float(phrase_values[indices].mean())
    return risk, phrase, task1_score(risk, phrase), phrase_values


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame = load_train_data().reset_index(drop=True)
    labels = frame.risk_label.to_numpy(); groups = frame.anon_user_id.astype(str).to_numpy()
    outer_train, _ = next(StratifiedGroupKFold(
        n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED
    ).split(np.zeros(len(frame)), labels, groups))
    records, membership_map, _ = _load_records()
    global_indices = np.asarray([int(row["global_index"]) for row in records])
    membership = np.asarray([membership_map[int(index)] for index in global_indices])
    truth = labels[global_indices]; texts = [row["text"] for row in records]
    local_groups = groups[global_indices]
    transformer = np.vstack([row["old_probability"] for row in records])
    ordinal_decisions = _train_decisions(frame, outer_train, records, membership_map)
    v36_saved = np.load(V36_CACHE, allow_pickle=True)
    names = v36_saved["names"].tolist(); nominal = v36_saved["decisions"][names.index("svc-c0.5-balanced")]
    evidence = _evidence_matrix(records)
    baseline_parameters = {"temperature": 1., "ordinal_weight": 0.}
    baseline = _predict(texts, transformer, nominal, ordinal_decisions[0], baseline_parameters)
    base_metric = _metric(truth, baseline, evidence, np.arange(len(truth)))

    crossfit = np.zeros(len(truth), dtype=np.int64); folds = []; choices = []
    grid = [(c_index, temperature, weight) for c_index in range(len(C_VALUES))
            for temperature in (.5, .7, 1., 1.3, 1.6)
            for weight in (0., .1, .2, .3, .4, .5)]
    for fold in range(4):
        fit = np.flatnonzero(membership != fold); held = np.flatnonzero(membership == fold)
        candidates = []
        for c_index, temperature, weight in grid:
            parameters = {"c": C_VALUES[c_index], "temperature": temperature,
                          "ordinal_weight": weight}
            prediction = _predict(texts, transformer, nominal,
                                  ordinal_decisions[c_index], parameters)
            metric = _metric(truth, prediction, evidence, fit)
            candidates.append((metric[2], metric[0], parameters, prediction))
        _, _, selected, prediction = max(candidates, key=lambda row: (row[0], row[1]))
        crossfit[held] = prediction[held]; choices.append(selected)
        old = _metric(truth, baseline, evidence, held); new = _metric(truth, prediction, evidence, held)
        folds.append({"fold": fold, "posts": int(len(held)), **selected,
                      "baseline_risk_f1": old[0], "candidate_risk_f1": new[0],
                      "baseline_task1": old[2], "candidate_task1": new[2]})
        print(f"V38 fold={fold} risk {old[0]:.6f} -> {new[0]:.6f}, "
              f"task1 {old[2]:.6f} -> {new[2]:.6f}", flush=True)

    cross_metric = _metric(truth, crossfit, evidence, np.arange(len(truth)))
    from collections import Counter
    production = {key: Counter(row[key] for row in choices).most_common(1)[0][0]
                  for key in ("c", "temperature", "ordinal_weight")}
    c_index = C_VALUES.index(float(production["c"]))
    fixed = _predict(texts, transformer, nominal, ordinal_decisions[c_index], production)
    fixed_metric = _metric(truth, fixed, evidence, np.arange(len(truth)))
    unique = np.unique(local_groups); rng = np.random.default_rng(config.SEED + 3838); deltas = []
    for _ in range(4000):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        pos = np.concatenate([np.flatnonzero(local_groups == user) for user in sampled])
        old_f1 = f1_score(truth[pos], baseline[pos], average="weighted", zero_division=0)
        new_f1 = f1_score(truth[pos], crossfit[pos], average="weighted", zero_division=0)
        deltas.append(task1_score(new_f1, float(cross_metric[3][pos].mean()))
                      - task1_score(old_f1, float(base_metric[3][pos].mean())))
    deltas = np.asarray(deltas)
    bootstrap = {"mean_task1_delta": float(deltas.mean()),
                 "p05_task1_delta": float(np.quantile(deltas, .05)),
                 "p95_task1_delta": float(np.quantile(deltas, .95)),
                 "positive_fraction": float((deltas > 0).mean())}
    adopted = bool(cross_metric[2] >= base_metric[2] + .003
                   and fixed_metric[2] >= base_metric[2] + .003
                   and bootstrap["positive_fraction"] >= .80)
    if adopted:
        vectorizer = _vectorizer(); matrix = vectorizer.fit_transform(frame.text.astype(str))
        models = []
        for threshold in range(3):
            target = (labels > threshold).astype(np.int64)
            models.append(LinearSVC(C=float(production["c"]), class_weight="balanced")
                          .fit(matrix, target))
        joblib.dump({"training_version": TRAINING_VERSION, "vectorizer": vectorizer,
                     "models": models}, ARTIFACT)
    payload = {"training_version": TRAINING_VERSION,
        "evaluation_scope": "four-fold user-disjoint ordinal lexical OOF",
        "baseline_v36": {"risk_f1": base_metric[0], "phrase_f1": base_metric[1],
                         "task1": base_metric[2]},
        "crossfit_candidate": {"risk_f1": cross_metric[0], "phrase_f1": cross_metric[1],
                               "task1": cross_metric[2], "folds": folds},
        "fixed_production": {**production, "risk_f1": fixed_metric[0],
                             "phrase_f1": fixed_metric[1], "task1": fixed_metric[2]},
        "user_cluster_bootstrap": bootstrap, "adopted": adopted}
    RESULTS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    CALIBRATION.write_text(json.dumps({"training_version": TRAINING_VERSION,
        "adopted": adopted, **production, "crossfit_task1": cross_metric[2]},
        indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True); return payload


if __name__ == "__main__":
    main()
