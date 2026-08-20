"""Nested selective expert gate for V36 disagreements (V43)."""
from __future__ import annotations

from collections import Counter
import json

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC

from analyze_task1_lexical_v11 import _softmax
from analyze_task1_oof_risk_v36 import CACHE as V36_CACHE, _evidence_matrix, _predict
from baseline import _vectorizer
from configs.config import config
from inference.task1_evidence_v4 import correct_risk_only
from preprocess.preprocess import load_train_data
from trainer.task1_atomic_v25 import _load_records
from utils.task1_metric import task1_score


OUTPUT = config.OUTPUT_DIR / "task1_selective_gate_v43"
RESULTS = OUTPUT / "results.json"
CALIBRATION = OUTPUT / "calibration.json"
TRAINING_VERSION = "task1-nested-selective-expert-gate-v43"


def _metric(truth, prediction, evidence, indices):
    indices = np.asarray(indices, dtype=np.int64)
    risk = float(f1_score(truth[indices], prediction[indices], average="weighted",
                          zero_division=0))
    phrase_values = evidence[np.arange(len(prediction)), prediction]
    phrase = float(phrase_values[indices].mean())
    return risk, phrase, task1_score(risk, phrase), phrase_values


def _numeric(old_probability, lexical_probability, base, raw, lexical, texts):
    def one_hot(values):
        return np.eye(4, dtype=np.float32)[values]
    def margins(probability):
        ordered = np.sort(probability, axis=1)
        return (ordered[:, -1] - ordered[:, -2])[:, None]
    lengths = np.asarray([[np.log1p(len(text)), np.log1p(len(text.split()))]
                          for text in texts], dtype=np.float32)
    return np.hstack((old_probability, lexical_probability,
                      one_hot(base), one_hot(raw), one_hot(lexical),
                      margins(old_probability), margins(lexical_probability), lengths))


def _fit_gate(matrix, truth, base, alternative, indices, c_value, balanced):
    indices = np.asarray(indices, dtype=np.int64)
    disagreement = indices[base[indices] != alternative[indices]]
    informative = disagreement[(base[disagreement] == truth[disagreement])
                               | (alternative[disagreement] == truth[disagreement])]
    targets = (alternative[informative] == truth[informative]).astype(np.int64)
    if len(informative) < 10 or len(np.unique(targets)) < 2:
        return None
    model = LinearSVC(C=float(c_value), class_weight="balanced" if balanced else None)
    model.fit(matrix[informative], targets)
    return model


def _apply_gate(model, matrix, base, alternative, indices, threshold):
    indices = np.asarray(indices, dtype=np.int64)
    result = base.copy()
    disagreement = indices[base[indices] != alternative[indices]]
    if model is None or not len(disagreement):
        return result
    scores = model.decision_function(matrix[disagreement])
    selected = disagreement[np.asarray(scores) >= float(threshold)]
    result[selected] = alternative[selected]
    return result


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame = load_train_data().reset_index(drop=True)
    labels = frame.risk_label.to_numpy(dtype=np.int64)
    groups = frame.anon_user_id.astype(str).to_numpy()
    records, membership_map, _ = _load_records()
    global_indices = np.asarray([int(row["global_index"]) for row in records])
    truth = labels[global_indices]
    local_groups = groups[global_indices]
    membership = np.asarray([membership_map[int(index)] for index in global_indices])
    texts = [str(row["text"]) for row in records]

    saved = np.load(V36_CACHE, allow_pickle=True)
    names = saved["names"].tolist(); decisions = saved["decisions"]
    parameters = json.loads((config.OUTPUT_DIR / "task1_oof_risk_v36" / "calibration.json")
                            .read_text(encoding="utf-8"))
    decision = decisions[names.index(parameters["expert"])]
    old_probability = np.vstack([row["old_probability"] for row in records])
    lexical_probability = _softmax(decision, parameters["temperature"])
    corrections = np.asarray([[correct_risk_only(text, risk) for risk in range(4)]
                              for text in texts], dtype=np.int64)
    base = _predict(old_probability, decision, parameters, corrections)
    raw = corrections[np.arange(len(truth)), old_probability.argmax(1)]
    lexical = corrections[np.arange(len(truth)), decision.argmax(1)]
    alternatives = {"raw_transformer": raw, "lexical": lexical}
    evidence = _evidence_matrix(records)
    baseline = _metric(truth, base, evidence, np.arange(len(truth)))
    numeric = _numeric(old_probability, lexical_probability, base, raw, lexical, texts)

    c_values = (.01, .03, .1, .3)
    thresholds = (0., .25, .50, .75, 1.)
    crossfit = np.zeros(len(truth), dtype=np.int64); selections = []; folds = []
    for outer_fold in range(4):
        outer_fit = np.flatnonzero(membership != outer_fold)
        outer_held = np.flatnonzero(membership == outer_fold)
        vectorizer = _vectorizer()
        text_fit = vectorizer.fit_transform([texts[index] for index in outer_fit])
        # Transform all rows once; only labels from outer_fit are consumed.
        text_all = vectorizer.transform(texts)
        matrix = hstack((text_all, csr_matrix(numeric * 2.0)), format="csr")
        candidate_rows = []
        fit_folds = [fold for fold in range(4) if fold != outer_fold]
        for alternative_name, alternative in alternatives.items():
            for c_value in c_values:
                for balanced in (False, True):
                    inner_scores = {threshold: base.copy() for threshold in thresholds}
                    for inner_fold in fit_folds:
                        inner_train = np.flatnonzero((membership != outer_fold)
                                                     & (membership != inner_fold))
                        inner_held = np.flatnonzero(membership == inner_fold)
                        model = _fit_gate(matrix, truth, base, alternative, inner_train,
                                          c_value, balanced)
                        for threshold in thresholds:
                            prediction = _apply_gate(model, matrix, base, alternative,
                                                     inner_held, threshold)
                            inner_scores[threshold][inner_held] = prediction[inner_held]
                    for threshold in thresholds:
                        score = _metric(truth, inner_scores[threshold], evidence, outer_fit)
                        candidate_rows.append((score[2], score[0], alternative_name,
                                               c_value, balanced, threshold))
        _, _, alternative_name, c_value, balanced, threshold = max(
            candidate_rows, key=lambda row: (row[0], row[1])
        )
        alternative = alternatives[alternative_name]
        gate = _fit_gate(matrix, truth, base, alternative, outer_fit, c_value, balanced)
        prediction = _apply_gate(gate, matrix, base, alternative, outer_held, threshold)
        crossfit[outer_held] = prediction[outer_held]
        old = _metric(truth, base, evidence, outer_held)
        new = _metric(truth, crossfit, evidence, outer_held)
        changes = int((prediction[outer_held] != base[outer_held]).sum())
        selections.append((alternative_name, c_value, balanced, threshold))
        folds.append({"fold": outer_fold, "posts": int(len(outer_held)),
                      "alternative": alternative_name, "c": c_value,
                      "balanced": balanced, "threshold": threshold,
                      "changed_predictions": changes,
                      "baseline_risk_f1": old[0], "candidate_risk_f1": new[0],
                      "baseline_task1": old[2], "candidate_task1": new[2]})
        print(f"V43 fold={outer_fold} alt={alternative_name} C={c_value} "
              f"thr={threshold} changes={changes} task1 {old[2]:.6f}->{new[2]:.6f}",
              flush=True)

    candidate = _metric(truth, crossfit, evidence, np.arange(len(truth)))
    unique = np.unique(local_groups); rng = np.random.default_rng(config.SEED + 4343)
    deltas = []
    for _ in range(4000):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        selected = np.concatenate([np.flatnonzero(local_groups == user) for user in sampled])
        old_risk = f1_score(truth[selected], base[selected], average="weighted",
                            zero_division=0)
        new_risk = f1_score(truth[selected], crossfit[selected], average="weighted",
                            zero_division=0)
        deltas.append(task1_score(new_risk, float(candidate[3][selected].mean()))
                      - task1_score(old_risk, float(baseline[3][selected].mean())))
    deltas = np.asarray(deltas)
    bootstrap = {"mean_task1_delta": float(deltas.mean()),
                 "p05_task1_delta": float(np.quantile(deltas, .05)),
                 "p95_task1_delta": float(np.quantile(deltas, .95)),
                 "positive_fraction": float((deltas > 0).mean())}
    non_worse = sum(row["candidate_task1"] >= row["baseline_task1"] for row in folds)
    adopted = bool(candidate[2] >= baseline[2] + .003
                   and bootstrap["positive_fraction"] >= .80 and non_worse >= 3)
    production = Counter(selections).most_common(1)[0][0]
    payload = {"training_version": TRAINING_VERSION,
               "evaluation_scope": "four outer user folds with nested three-fold gate selection",
               "baseline_v36": {"risk_f1": baseline[0], "phrase_f1": baseline[1],
                                "task1": baseline[2]},
               "crossfit_candidate": {"risk_f1": candidate[0], "phrase_f1": candidate[1],
                                      "task1": candidate[2], "folds": folds},
               "production_parameter_mode": {
                   "alternative": production[0], "c": production[1],
                   "balanced": production[2], "threshold": production[3]},
               "user_cluster_bootstrap": bootstrap, "adopted": adopted}
    RESULTS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    CALIBRATION.write_text(json.dumps({"training_version": TRAINING_VERSION,
        "adopted": adopted, "alternative": production[0], "c": production[1],
        "balanced": production[2], "threshold": production[3],
        "crossfit_task1": candidate[2], "baseline_task1": baseline[2]}, indent=2),
        encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    return payload


if __name__ == "__main__":
    main()
