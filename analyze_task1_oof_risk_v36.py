"""Full nested-OOF calibration of the V18 Task 1 risk fusion (V36)."""
from __future__ import annotations

from collections import Counter
import json

import joblib
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.svm import LinearSVC

from analyze_task1_lexical_v11 import _lexical_experts, _softmax
from baseline import _post_phrase_f1, _vectorizer
from configs.config import config
from inference.task1_evidence_v4 import (
    apply_evidence_policy, correct_risk_only, decode_model_evidence,
)
from preprocess.preprocess import load_train_data
from trainer.task1_atomic_v25 import _load_records
from utils.task1_metric import task1_score


OUTPUT = config.OUTPUT_DIR / "task1_oof_risk_v36"
RESULTS = OUTPUT / "results.json"
CALIBRATION = OUTPUT / "calibration.json"
CACHE = OUTPUT / "oof_lexical_decisions.npz"
FULL_LEXICAL_MODEL = OUTPUT / "full_lexical_svc.joblib"
TRAINING_VERSION = "task1-full-oof-risk-calibration-v36"


def _lexical_oof(frame, outer_train, records, membership):
    expert_names = None
    if CACHE.exists():
        saved = np.load(CACHE, allow_pickle=True)
        if np.array_equal(saved["global_indices"], np.asarray(
                [int(row["global_index"]) for row in records])):
            print("V36 resumed OOF lexical decisions", flush=True)
            return saved["names"].tolist(), saved["decisions"]
    global_indices = np.asarray([int(row["global_index"]) for row in records])
    local = {index: position for position, index in enumerate(global_indices)}
    decisions_by_name = {}
    for fold in range(4):
        fit = np.asarray([i for i in outer_train if membership[int(i)] != fold])
        valid = np.asarray([i for i in outer_train if membership[int(i)] == fold])
        experts = _lexical_experts(frame, fit, valid)
        if expert_names is None:
            expert_names = list(experts)
            decisions_by_name = {name: np.zeros((len(records), 4), dtype=np.float32)
                                 for name in expert_names}
        for name in expert_names:
            decisions_by_name[name][[local[int(i)] for i in valid]] = experts[name]
        print(f"V36 lexical OOF fold {fold + 1}/4", flush=True)
    decisions = np.stack([decisions_by_name[name] for name in expert_names])
    names = np.empty(len(expert_names), dtype=object); names[:] = expert_names
    np.savez_compressed(CACHE, global_indices=global_indices,
                        names=names, decisions=decisions)
    return expert_names, decisions


def _evidence_matrix(records):
    calibration = json.loads((
        config.OUTPUT_DIR / "task1_oof_decoder_v35" / "calibration.json"
    ).read_text(encoding="utf-8"))
    recipes = calibration["parameters_by_predicted_risk"]
    matrix = np.zeros((len(records), 4), dtype=np.float32)
    for risk in range(4):
        parameters = recipes[config.ID2RISK[risk]]
        for index, record in enumerate(records):
            spans = decode_model_evidence(
                record["text"], record["offsets"], record["start"], record["end"],
                threshold=float(parameters["threshold"]),
                max_tokens=int(parameters["max_tokens"]),
                end_policy=str(parameters["end_policy"]), limit=5,
            )
            evidence = apply_evidence_policy(
                record["text"], risk, spans,
                policy=str(parameters["cue_policy"]), topk=int(parameters["topk"]),
            )
            matrix[index, risk] = _post_phrase_f1(evidence, record["gold"])
    return matrix


def _predict(transformer, decision, parameters, corrections):
    lexical = _softmax(decision, parameters["temperature"])
    probability = ((1. - parameters["lexical_weight"]) * transformer
                   + parameters["lexical_weight"] * lexical)
    logits = np.log(np.clip(probability, 1e-8, 1.))
    logits[:, 0] += parameters["indicator_bias"]
    logits[:, 2] += parameters["behavior_bias"]
    logits[:, 3] += parameters["attempt_bias"]
    raw = logits.argmax(1)
    return corrections[np.arange(len(raw)), raw]


def _metric(truth, prediction, evidence_matrix, indices):
    indices = np.asarray(indices, dtype=np.int64)
    risk = float(f1_score(truth[indices], prediction[indices],
                          average="weighted", zero_division=0))
    phrase_values = evidence_matrix[np.arange(len(prediction)), prediction]
    phrase = float(phrase_values[indices].mean())
    return risk, phrase, task1_score(risk, phrase), phrase_values


def _grid(truth, transformer, names, decisions, evidence_matrix, fit, corrections):
    rows = []
    for expert_index, name in enumerate(names):
        for temperature in (.5, .7, 1., 1.3):
            for lexical_weight in (.0, .2, .3, .4, .5, .6, .7):
                if lexical_weight == 0 and (expert_index > 0 or temperature != .5):
                    continue
                for indicator_bias in (-.15, 0., .15):
                    for behavior_bias in (-.20, 0., .20):
                        for attempt_bias in (0., .20, .40):
                            parameters = {"expert": name, "temperature": temperature,
                                "lexical_weight": lexical_weight,
                                "indicator_bias": indicator_bias,
                                "behavior_bias": behavior_bias,
                                "attempt_bias": attempt_bias}
                            prediction = _predict(
                                transformer, decisions[expert_index], parameters, corrections
                            )
                            risk, phrase, task, _ = _metric(
                                truth, prediction, evidence_matrix, fit
                            )
                            rows.append({**parameters, "risk_f1": risk,
                                         "phrase_f1": phrase, "task1": task})
    return sorted(rows, key=lambda row: (row["task1"], row["risk_f1"]), reverse=True)


def _aggregate(rows):
    def mode(values): return Counter(values).most_common(1)[0][0]
    def median(values):
        centre = float(np.median(values))
        return min(values, key=lambda value: (abs(float(value) - centre), float(value)))
    return {"expert": mode([row["expert"] for row in rows]),
            "temperature": float(median([row["temperature"] for row in rows])),
            "lexical_weight": float(median([row["lexical_weight"] for row in rows])),
            "indicator_bias": float(median([row["indicator_bias"] for row in rows])),
            "behavior_bias": float(median([row["behavior_bias"] for row in rows])),
            "attempt_bias": float(median([row["attempt_bias"] for row in rows]))}


def _fit_full_lexical(frame, parameters):
    expert = str(parameters["expert"])
    match = __import__("re").fullmatch(r"svc-c([0-9.]+)-(balanced|plain)", expert)
    if match is None:
        raise ValueError(f"V36 production expert is not an SVC: {expert}")
    c_value = float(match.group(1)); class_weight = (
        "balanced" if match.group(2) == "balanced" else None
    )
    vectorizer = _vectorizer(); matrix = vectorizer.fit_transform(frame.text.astype(str))
    model = LinearSVC(C=c_value, class_weight=class_weight).fit(matrix, frame.risk_label)
    joblib.dump({"training_version": TRAINING_VERSION, "expert": expert,
                 "train_posts": int(len(frame)), "vectorizer": vectorizer,
                 "risk_model": model}, FULL_LEXICAL_MODEL)
    print(f"V36 fitted full lexical expert: {FULL_LEXICAL_MODEL}", flush=True)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame = load_train_data().reset_index(drop=True)
    labels = frame.risk_label.to_numpy(); groups = frame.anon_user_id.astype(str).to_numpy()
    outer_train, _ = next(StratifiedGroupKFold(
        n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED
    ).split(np.zeros(len(frame)), labels, groups))
    records, membership, _ = _load_records()
    global_indices = np.asarray([int(row["global_index"]) for row in records])
    truth = labels[global_indices]; local_groups = groups[global_indices]
    transformer = np.vstack([row["old_probability"] for row in records])
    # Definition-aligned cue correction depends only on (post, raw class).
    # Precompute the 1,305 x 4 lookup once instead of running regexes tens of
    # millions of times inside the calibration grid.
    corrections = np.asarray([
        [correct_risk_only(row["text"], risk) for risk in range(4)]
        for row in records
    ], dtype=np.int64)
    names, decisions = _lexical_oof(frame, outer_train, records, membership)
    evidence_matrix = _evidence_matrix(records)
    name_to_index = {name: index for index, name in enumerate(names)}

    deployed_parameters = {"expert": "svc-c0.25-balanced", "temperature": 1.,
        "lexical_weight": .6, "indicator_bias": 0., "behavior_bias": 0.,
        "attempt_bias": .2}
    deployed = _predict(transformer,
                        decisions[name_to_index[deployed_parameters["expert"]]],
                        deployed_parameters, corrections)
    deployed_risk, deployed_phrase, deployed_task, deployed_phrase_values = _metric(
        truth, deployed, evidence_matrix, np.arange(len(records))
    )

    crossfit = np.zeros(len(records), dtype=np.int64); selections = []; fold_rows = []
    for fold in range(4):
        fit = np.flatnonzero(np.asarray([membership[int(i)] for i in global_indices]) != fold)
        held = np.flatnonzero(np.asarray([membership[int(i)] for i in global_indices]) == fold)
        selected = _grid(truth, transformer, names, decisions,
                         evidence_matrix, fit, corrections)[0]
        parameters = {key: selected[key] for key in (
            "expert", "temperature", "lexical_weight", "indicator_bias",
            "behavior_bias", "attempt_bias")}
        prediction = _predict(transformer,
                              decisions[name_to_index[parameters["expert"]]],
                              parameters, corrections)
        crossfit[held] = prediction[held]
        old_risk, old_phrase, old_task, _ = _metric(
            truth, deployed, evidence_matrix, held)
        new_risk, new_phrase, new_task, _ = _metric(
            truth, prediction, evidence_matrix, held)
        fold_rows.append({"fold": fold, "posts": int(len(held)), **parameters,
                          "deployed_risk_f1": old_risk,
                          "candidate_risk_f1": new_risk,
                          "deployed_task1": old_task, "candidate_task1": new_task})
        selections.append(parameters)
        print(f"V36 fold={fold} risk {old_risk:.6f} -> {new_risk:.6f}, "
              f"task1 {old_task:.6f} -> {new_task:.6f}", flush=True)

    cross_risk, cross_phrase, cross_task, cross_phrase_values = _metric(
        truth, crossfit, evidence_matrix, np.arange(len(records))
    )
    production = _aggregate(selections)
    fixed = _predict(transformer,
                     decisions[name_to_index[production["expert"]]],
                     production, corrections)
    fixed_risk, fixed_phrase, fixed_task, _ = _metric(
        truth, fixed, evidence_matrix, np.arange(len(records))
    )
    unique = np.unique(local_groups); rng = np.random.default_rng(config.SEED + 3636); deltas = []
    for _ in range(4000):
        sampled_users = rng.choice(unique, size=len(unique), replace=True)
        positions = np.concatenate([np.flatnonzero(local_groups == user) for user in sampled_users])
        old_risk = f1_score(truth[positions], deployed[positions],
                            average="weighted", zero_division=0)
        new_risk = f1_score(truth[positions], crossfit[positions],
                            average="weighted", zero_division=0)
        deltas.append(task1_score(new_risk, float(cross_phrase_values[positions].mean()))
                      - task1_score(old_risk, float(deployed_phrase_values[positions].mean())))
    deltas = np.asarray(deltas)
    bootstrap = {"mean_task1_delta": float(deltas.mean()),
                 "p05_task1_delta": float(np.quantile(deltas, .05)),
                 "p95_task1_delta": float(np.quantile(deltas, .95)),
                 "positive_fraction": float((deltas > 0).mean())}
    adopted = bool(cross_task >= deployed_task + .003
                   and fixed_task >= deployed_task + .003
                   and bootstrap["positive_fraction"] >= .80)
    payload = {"training_version": TRAINING_VERSION,
        "evaluation_scope": "all 1305 nested OOF posts; four-fold cross-fitted fusion",
        "deployed_v18_without_v2": {**deployed_parameters, "risk_f1": deployed_risk,
                                    "phrase_f1": deployed_phrase, "task1": deployed_task},
        "crossfit_candidate": {"risk_f1": cross_risk, "phrase_f1": cross_phrase,
                               "task1": cross_task, "folds": fold_rows},
        "fixed_production_diagnostic": {**production, "risk_f1": fixed_risk,
                                        "phrase_f1": fixed_phrase, "task1": fixed_task},
        "user_cluster_bootstrap": bootstrap, "adopted": adopted}
    RESULTS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    CALIBRATION.write_text(json.dumps({"training_version": TRAINING_VERSION,
        "adopted": adopted, **production, "crossfit_task1": cross_task,
        "deployed_task1": deployed_task}, indent=2), encoding="utf-8")
    if adopted:
        _fit_full_lexical(frame, production)
    print(json.dumps(payload, indent=2), flush=True); return payload


if __name__ == "__main__":
    main()
