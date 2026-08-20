"""Nested OOF logistic stacking for Task 1 risk (V37)."""
from __future__ import annotations

from collections import Counter
import json
import re

import joblib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from analyze_task1_lexical_v11 import _lexical_experts
from analyze_task1_oof_risk_v36 import (
    CACHE as V36_CACHE, _evidence_matrix, _predict as _v36_predict,
)
from baseline import _vectorizer
from configs.config import config
from inference.task1_evidence_v4 import (
    EXTENDED_ATTEMPT_CUE, EXTENDED_BEHAVIOR_CUE, EXTENDED_IDEATION_CUE,
    correct_risk_only,
)
from preprocess.preprocess import load_train_data
from trainer.task1_atomic_v25 import _load_records
from utils.task1_metric import task1_score


OUTPUT = config.OUTPUT_DIR / "task1_oof_meta_v37"
RESULTS = OUTPUT / "results.json"
CALIBRATION = OUTPUT / "calibration.json"
ARTIFACT = OUTPUT / "full_meta.joblib"
TRAINING_VERSION = "task1-nested-oof-logistic-stack-v37"


def _sigmoid_statistics(values):
    probability = torch.sigmoid(values.float()).numpy().ravel()
    if not len(probability):
        return [0., 0., 0.]
    top = np.partition(probability, max(0, len(probability) - min(5, len(probability))))[-5:]
    return [float(probability.max()), float(top.mean()), float((probability >= .55).mean())]


def build_meta_features(texts, transformer, decisions, starts, ends):
    """Create distribution-stable features available in both OOF and test."""
    transformer = np.asarray(transformer, dtype=np.float64)
    decisions = np.asarray(decisions, dtype=np.float64)  # [experts, posts, classes]
    shifted = decisions - decisions.max(2, keepdims=True)
    lexical_probability = np.exp(shifted)
    lexical_probability /= lexical_probability.sum(2, keepdims=True)
    votes = np.stack([(decisions.argmax(2) == risk).mean(0) for risk in range(4)], axis=1)
    rows = []
    for index, text in enumerate(texts):
        value = str(text); lower = value.casefold()
        discourse = [
            float(bool(EXTENDED_IDEATION_CUE.search(value))),
            float(bool(EXTENDED_BEHAVIOR_CUE.search(value))),
            float(bool(EXTENDED_ATTEMPT_CUE.search(value))),
            float(bool(re.search(r"\b(?:i|i'm|i've|me|my|myself)\b", value, re.I))),
            float(bool(re.search(r"\b(?:not|never|don't|didn't|won't)\b", value, re.I))),
            np.log1p(len(value)), np.log1p(len(value.split())),
            float(lower.count("suicid")), float(lower.count("die")),
        ]
        row = np.concatenate([
            transformer[index], np.log(np.clip(transformer[index], 1e-7, 1.)),
            decisions[:, index].ravel(), lexical_probability[:, index].ravel(),
            votes[index], np.asarray(_sigmoid_statistics(starts[index])),
            np.asarray(_sigmoid_statistics(ends[index])), np.asarray(discourse),
        ])
        rows.append(row)
    return np.vstack(rows).astype(np.float32)


def _correct(texts, raw):
    return np.asarray([correct_risk_only(text, int(risk))
                       for text, risk in zip(texts, raw)], dtype=np.int64)


def _model(c_value, class_weight):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=float(c_value), class_weight=class_weight,
                           max_iter=3000, solver="lbfgs"),
    )


def _metric(truth, prediction, evidence, indices):
    indices = np.asarray(indices, dtype=np.int64)
    risk = float(f1_score(truth[indices], prediction[indices],
                          average="weighted", zero_division=0))
    phrase_values = evidence[np.arange(len(prediction)), prediction]
    phrase = float(phrase_values[indices].mean())
    return risk, phrase, task1_score(risk, phrase), phrase_values


def _select_nested(features, truth, texts, membership, evidence, outer_fold):
    fit_folds = [fold for fold in range(4) if fold != outer_fold]
    candidates = []
    for c_value in (.003, .01, .03, .1, .3, 1.):
        for class_weight in (None, "balanced"):
            values = []
            for inner_held in fit_folds:
                train = np.flatnonzero((membership != outer_fold) & (membership != inner_held))
                valid = np.flatnonzero(membership == inner_held)
                model = _model(c_value, class_weight).fit(features[train], truth[train])
                prediction = _correct(
                    [texts[index] for index in valid], model.predict(features[valid])
                )
                full = np.zeros(len(truth), dtype=np.int64); full[valid] = prediction
                risk, phrase, task, _ = _metric(truth, full, evidence, valid)
                values.append(task)
            candidates.append({"c": c_value, "class_weight": class_weight,
                               "inner_task1": float(np.mean(values))})
    return max(candidates, key=lambda row: row["inner_task1"])


def _fit_full_experts(frame, names):
    vectorizer = _vectorizer(); matrix = vectorizer.fit_transform(frame.text.astype(str))
    models = {}
    for name in names:
        match = re.fullmatch(r"svc-c([0-9.]+)-(balanced|plain)", name)
        c_value = float(match.group(1)); weight = "balanced" if match.group(2) == "balanced" else None
        models[name] = LinearSVC(C=c_value, class_weight=weight).fit(matrix, frame.risk_label)
    return vectorizer, models


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame = load_train_data().reset_index(drop=True)
    labels = frame.risk_label.to_numpy(); groups = frame.anon_user_id.astype(str).to_numpy()
    outer_train, outer_valid = next(StratifiedGroupKFold(
        n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED
    ).split(np.zeros(len(frame)), labels, groups))
    records, membership_map, outer_raw = _load_records()
    global_indices = np.asarray([int(row["global_index"]) for row in records])
    membership = np.asarray([membership_map[int(index)] for index in global_indices])
    saved = np.load(V36_CACHE, allow_pickle=True)
    all_names = saved["names"].tolist(); mask = [name.startswith("svc-") for name in all_names]
    names = [name for name, keep in zip(all_names, mask) if keep]
    decisions = saved["decisions"][mask]
    transformer = np.vstack([row["old_probability"] for row in records])
    texts = [row["text"] for row in records]
    features = build_meta_features(
        texts, transformer, decisions,
        [row["start"] for row in records], [row["end"] for row in records],
    )
    truth = labels[global_indices]; local_groups = groups[global_indices]
    evidence = _evidence_matrix(records)

    v36 = json.loads((config.OUTPUT_DIR / "task1_oof_risk_v36" / "calibration.json")
                     .read_text(encoding="utf-8"))
    name_to_index = {name: index for index, name in enumerate(names)}
    corrections = np.asarray([[correct_risk_only(text, risk) for risk in range(4)]
                              for text in texts], dtype=np.int64)
    baseline = _v36_predict(
        transformer, decisions[name_to_index[v36["expert"]]], v36, corrections
    )
    base_risk, base_phrase, base_task, base_phrase_values = _metric(
        truth, baseline, evidence, np.arange(len(truth))
    )

    crossfit = np.zeros(len(truth), dtype=np.int64); selections = []; folds = []
    for fold in range(4):
        selected = _select_nested(features, truth, texts, membership, evidence, fold)
        fit = np.flatnonzero(membership != fold); held = np.flatnonzero(membership == fold)
        model = _model(selected["c"], selected["class_weight"]).fit(features[fit], truth[fit])
        prediction = _correct([texts[index] for index in held], model.predict(features[held]))
        crossfit[held] = prediction
        old = _metric(truth, baseline, evidence, held); new = _metric(truth, crossfit, evidence, held)
        folds.append({"fold": fold, "posts": int(len(held)), **selected,
                      "baseline_risk_f1": old[0], "candidate_risk_f1": new[0],
                      "baseline_task1": old[2], "candidate_task1": new[2]})
        selections.append(selected)
        print(f"V37 fold={fold} risk {old[0]:.6f} -> {new[0]:.6f}, "
              f"task1 {old[2]:.6f} -> {new[2]:.6f}", flush=True)

    cross_risk, cross_phrase, cross_task, cross_phrase_values = _metric(
        truth, crossfit, evidence, np.arange(len(truth))
    )
    chosen_c = Counter(row["c"] for row in selections).most_common(1)[0][0]
    chosen_weight = Counter(str(row["class_weight"]) for row in selections).most_common(1)[0][0]
    chosen_weight = None if chosen_weight == "None" else chosen_weight
    fixed_model = _model(chosen_c, chosen_weight).fit(features, truth)
    fixed = _correct(texts, fixed_model.predict(features))
    fixed_metric = _metric(truth, fixed, evidence, np.arange(len(truth)))

    unique = np.unique(local_groups); rng = np.random.default_rng(config.SEED + 3737); deltas = []
    for _ in range(4000):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        pos = np.concatenate([np.flatnonzero(local_groups == user) for user in sampled])
        old_f1 = f1_score(truth[pos], baseline[pos], average="weighted", zero_division=0)
        new_f1 = f1_score(truth[pos], crossfit[pos], average="weighted", zero_division=0)
        deltas.append(task1_score(new_f1, float(cross_phrase_values[pos].mean()))
                      - task1_score(old_f1, float(base_phrase_values[pos].mean())))
    deltas = np.asarray(deltas)
    bootstrap = {"mean_task1_delta": float(deltas.mean()),
                 "p05_task1_delta": float(np.quantile(deltas, .05)),
                 "p95_task1_delta": float(np.quantile(deltas, .95)),
                 "positive_fraction": float((deltas > 0).mean())}
    adopted = bool(cross_task >= base_task + .004
                   and bootstrap["positive_fraction"] >= .80)

    if adopted:
        # Add the clean outer holdout as another OOF block before the final
        # stacker fit, so all 1,635 labels contribute without in-sample base predictions.
        outer_records = outer_raw["records"]
        outer_experts = _lexical_experts(frame, outer_train, outer_valid)
        outer_decisions = np.stack([outer_experts[name] for name in names])
        outer_features = build_meta_features(
            [row["text"] for row in outer_records],
            np.vstack([row["old_probability"] for row in outer_records]), outer_decisions,
            [row["start"] for row in outer_records], [row["end"] for row in outer_records],
        )
        complete_features = np.vstack([features, outer_features])
        complete_truth = np.concatenate([truth, labels[outer_valid]])
        final_meta = _model(chosen_c, chosen_weight).fit(complete_features, complete_truth)
        vectorizer, expert_models = _fit_full_experts(frame, names)
        joblib.dump({"training_version": TRAINING_VERSION, "names": names,
                     "meta_model": final_meta, "vectorizer": vectorizer,
                     "expert_models": expert_models}, ARTIFACT)
        print(f"V37 full OOF stack artifact: {ARTIFACT}", flush=True)

    payload = {"training_version": TRAINING_VERSION,
        "evaluation_scope": "nested user-disjoint selection on 1305 OOF posts",
        "baseline_v36": {"risk_f1": base_risk, "phrase_f1": base_phrase, "task1": base_task},
        "crossfit_candidate": {"risk_f1": cross_risk, "phrase_f1": cross_phrase,
                               "task1": cross_task, "folds": folds},
        "fixed_in_sample_diagnostic": {"c": chosen_c, "class_weight": chosen_weight,
            "risk_f1": fixed_metric[0], "phrase_f1": fixed_metric[1], "task1": fixed_metric[2]},
        "user_cluster_bootstrap": bootstrap, "adopted": adopted}
    RESULTS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    CALIBRATION.write_text(json.dumps({"training_version": TRAINING_VERSION,
        "adopted": adopted, "c": chosen_c, "class_weight": chosen_weight,
        "crossfit_task1": cross_task}, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True); return payload


if __name__ == "__main__":
    main()
