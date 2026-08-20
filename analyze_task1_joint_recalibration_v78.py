"""Nested recalibration of the V77 shared hierarchy probabilities.

V77 changes the probability distribution of the first-stage neural model.  A
calibrator selected for the older V20 distribution is therefore not a fair
test.  V78 selects both the neural recipe and all lexical calibration
parameters using only the other three user-disjoint folds, then evaluates on
the held fold.  No further neural training is required.
"""
from __future__ import annotations

from collections import Counter
import json

import numpy as np
import torch
from sklearn.metrics import f1_score

from analyze_task1_oof_risk_v36 import (
    CACHE as V36_CACHE, _aggregate, _evidence_matrix, _metric, _predict,
)
from analyze_task1_lexical_v11 import _softmax
from configs.config import config
from inference.task1_evidence_v4 import correct_risk_only
from preprocess.preprocess import load_train_data
from trainer.task1_atomic_v25 import _load_records
from utils.task1_metric import task1_score


OUTPUT = config.OUTPUT_DIR / "task1_joint_recalibration_v78"
RESULTS = OUTPUT / "results.json"
CALIBRATION = OUTPUT / "calibration.json"
V77 = config.OUTPUT_DIR / "task1_joint_hard_negative_v77"
TRAINING_VERSION = "task1-joint-nested-recalibration-v78"


def _load_candidate(records, global_indices):
    position = {int(index): place for place, index in enumerate(global_indices)}
    nominal = np.zeros((len(records), 4), dtype=np.float32)
    hierarchy = np.zeros_like(nominal)
    candidate_records = [dict(row) for row in records]
    for fold in range(4):
        saved = torch.load(V77 / f"fold{fold}_raw.pt", map_location="cpu",
                           weights_only=False)
        if saved.get("training_version") != "task1-shared-hierarchy-hard-negative-v77b":
            raise ValueError("V78 requires the finite-loss V77b folds")
        locations = [position[int(index)] for index in saved["global_indices"]]
        nominal[locations] = saved["nominal"]
        hierarchy[locations] = saved["hierarchy"]
        for local, location in enumerate(locations):
            candidate_records[location]["start"] = saved["start"][local]
            candidate_records[location]["end"] = saved["end"][local]
    return nominal, hierarchy, candidate_records


def _recipes(old, nominal, hierarchy):
    result = []
    # Keep a compact, predeclared grid.  The inner calibration already tunes
    # lexical weight/biases, so a very fine neural blend would only add noise.
    for hierarchy_mix in (0., .15, .30):
        continued = (1. - hierarchy_mix) * nominal + hierarchy_mix * hierarchy
        for new_weight in (.25, .50, .75, 1.):
            probability = (1. - new_weight) * old + new_weight * continued
            result.append((hierarchy_mix, new_weight, probability))
    return result


def _parameter_dict(row):
    return {key: row[key] for key in (
        "expert", "temperature", "lexical_weight", "indicator_bias",
        "behavior_bias", "attempt_bias",
    )}


def _weighted_f1_fast(truth, prediction):
    matrix = np.bincount(4 * truth + prediction, minlength=16).reshape(4, 4)
    true_positive = np.diag(matrix).astype(np.float64)
    support = matrix.sum(1).astype(np.float64)
    predicted = matrix.sum(0).astype(np.float64)
    denominator = support + predicted
    scores = np.divide(2. * true_positive, denominator,
                       out=np.zeros(4, dtype=np.float64), where=denominator > 0)
    return float((scores * support).sum() / max(support.sum(), 1.))


def _fast_grid(truth, transformer, names, decisions, evidence, fit, corrections):
    """Focused version of V36's grid with the same held-user discipline."""
    fit = np.asarray(fit, dtype=np.int64)
    expert_indices = [i for i, name in enumerate(names)
                      if name in {"svc-c0.25-balanced", "svc-c0.5-balanced",
                                  "svc-c1-balanced", "svc-c2-balanced"}]
    best = None
    row_index = np.arange(len(truth))
    for expert_index in expert_indices:
        for temperature in (.7, 1., 1.3):
            lexical = _softmax(decisions[expert_index], temperature)
            for lexical_weight in (.4, .5, .6, .7, .8):
                probability = ((1. - lexical_weight) * transformer
                               + lexical_weight * lexical)
                base_logits = np.log(np.clip(probability, 1e-8, 1.))
                for indicator_bias in (-.15, 0.):
                    for behavior_bias in (0., .2):
                        for attempt_bias in (.2, .4):
                            logits = base_logits.copy()
                            logits[:, 0] += indicator_bias
                            logits[:, 2] += behavior_bias
                            logits[:, 3] += attempt_bias
                            raw = logits.argmax(1)
                            prediction = corrections[row_index, raw]
                            risk = _weighted_f1_fast(truth[fit], prediction[fit])
                            phrase = float(evidence[row_index, prediction][fit].mean())
                            task = task1_score(risk, phrase)
                            row = {"expert": str(names[expert_index]),
                                   "temperature": temperature,
                                   "lexical_weight": lexical_weight,
                                   "indicator_bias": indicator_bias,
                                   "behavior_bias": behavior_bias,
                                   "attempt_bias": attempt_bias,
                                   "risk_f1": risk, "phrase_f1": phrase,
                                   "task1": task}
                            if best is None or (task, risk, phrase) > (
                                    best["task1"], best["risk_f1"], best["phrase_f1"]):
                                best = row
    return best


def _bootstrap(groups, truth, old_prediction, new_prediction,
               old_phrase, new_phrase, draws=4000):
    unique = np.unique(groups); rng = np.random.default_rng(config.SEED + 7878)
    values = []
    for _ in range(draws):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        positions = np.concatenate([np.flatnonzero(groups == user) for user in sampled])
        old_risk = f1_score(truth[positions], old_prediction[positions],
                            average="weighted", zero_division=0)
        new_risk = f1_score(truth[positions], new_prediction[positions],
                            average="weighted", zero_division=0)
        values.append(task1_score(new_risk, float(new_phrase[positions].mean()))
                      - task1_score(old_risk, float(old_phrase[positions].mean())))
    values = np.asarray(values)
    return {"mean_task1_delta": float(values.mean()),
            "p05_task1_delta": float(np.quantile(values, .05)),
            "p95_task1_delta": float(np.quantile(values, .95)),
            "positive_fraction": float((values > 0).mean())}


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame = load_train_data().reset_index(drop=True)
    labels = frame.risk_label.to_numpy(dtype=np.int64)
    groups = frame.anon_user_id.astype(str).to_numpy()
    records, membership_map, _ = _load_records()
    global_indices = np.asarray([int(row["global_index"]) for row in records])
    membership = np.asarray([membership_map[int(i)] for i in global_indices])
    truth = labels[global_indices]; local_groups = groups[global_indices]
    old = np.vstack([row["old_probability"] for row in records])
    nominal, hierarchy, candidate_records = _load_candidate(records, global_indices)
    old_evidence = _evidence_matrix(records)
    new_evidence = _evidence_matrix(candidate_records)
    evidence_sources = {"old": old_evidence, "continued": new_evidence}
    saved = np.load(V36_CACHE, allow_pickle=True)
    names = saved["names"].tolist(); decisions = saved["decisions"]
    name_to_index = {name: index for index, name in enumerate(names)}
    corrections = np.asarray([
        [correct_risk_only(row["text"], risk) for risk in range(4)] for row in records
    ], dtype=np.int64)
    recipes = _recipes(old, nominal, hierarchy)

    baseline_prediction = np.zeros(len(records), dtype=np.int64)
    baseline_phrase = np.zeros(len(records), dtype=np.float32)
    candidate_prediction = np.zeros(len(records), dtype=np.int64)
    candidate_phrase = np.zeros(len(records), dtype=np.float32)
    baseline_selections = []; candidate_selections = []; fold_rows = []
    for fold in range(4):
        fit = np.flatnonzero(membership != fold)
        held = np.flatnonzero(membership == fold)
        base_selected = _fast_grid(
            truth, old, names, decisions, old_evidence, fit, corrections
        )
        base_parameters = _parameter_dict(base_selected)
        base_fold = _predict(
            old, decisions[name_to_index[base_parameters["expert"]]],
            base_parameters, corrections,
        )
        baseline_prediction[held] = base_fold[held]
        base_phrase_values = old_evidence[np.arange(len(records)), base_fold]
        baseline_phrase[held] = base_phrase_values[held]
        baseline_selections.append(base_parameters)

        candidates = []
        for hierarchy_mix, new_weight, transformer in recipes:
            for evidence_name, evidence_matrix in evidence_sources.items():
                selected = _fast_grid(
                    truth, transformer, names, decisions, evidence_matrix,
                    fit, corrections,
                )
                candidates.append((selected["task1"], selected["risk_f1"],
                                   selected["phrase_f1"], -new_weight,
                                   hierarchy_mix, new_weight, evidence_name,
                                   transformer, selected, evidence_matrix))
        chosen = max(candidates, key=lambda row: row[:4])
        _, _, _, _, hierarchy_mix, new_weight, evidence_name, transformer, selected, evidence_matrix = chosen
        parameters = _parameter_dict(selected)
        predicted = _predict(
            transformer, decisions[name_to_index[parameters["expert"]]],
            parameters, corrections,
        )
        candidate_prediction[held] = predicted[held]
        phrase_values = evidence_matrix[np.arange(len(records)), predicted]
        candidate_phrase[held] = phrase_values[held]
        candidate_selections.append({
            "hierarchy_mix": hierarchy_mix, "new_model_weight": new_weight,
            "evidence_source": evidence_name, **parameters,
        })
        base_risk = f1_score(truth[held], base_fold[held], average="weighted",
                             zero_division=0)
        new_risk = f1_score(truth[held], predicted[held], average="weighted",
                            zero_division=0)
        base_task = task1_score(base_risk, float(base_phrase_values[held].mean()))
        new_task = task1_score(new_risk, float(phrase_values[held].mean()))
        fold_rows.append({"fold": fold, "posts": int(len(held)),
                          **candidate_selections[-1],
                          "baseline_task1": base_task, "candidate_task1": new_task})
        print(f"V78 fold={fold} hierarchy={hierarchy_mix:.2f} new={new_weight:.2f} "
              f"evidence={evidence_name} task1 {base_task:.6f}->{new_task:.6f}",
              flush=True)

    base_risk = float(f1_score(truth, baseline_prediction, average="weighted",
                               zero_division=0))
    new_risk = float(f1_score(truth, candidate_prediction, average="weighted",
                              zero_division=0))
    base_phrase_f1 = float(baseline_phrase.mean())
    new_phrase_f1 = float(candidate_phrase.mean())
    base_task = task1_score(base_risk, base_phrase_f1)
    new_task = task1_score(new_risk, new_phrase_f1)
    bootstrap = _bootstrap(local_groups, truth, baseline_prediction,
                           candidate_prediction, baseline_phrase, candidate_phrase)

    # Fixed diagnostic: aggregate nested choices without consulting any held
    # fold, just as V36 does for production parameters.
    hm = Counter(row["hierarchy_mix"] for row in candidate_selections).most_common(1)[0][0]
    nw = Counter(row["new_model_weight"] for row in candidate_selections).most_common(1)[0][0]
    es = Counter(row["evidence_source"] for row in candidate_selections).most_common(1)[0][0]
    parameters = _aggregate(candidate_selections)
    continued = (1. - hm) * nominal + hm * hierarchy
    transformer = (1. - nw) * old + nw * continued
    fixed_prediction = _predict(
        transformer, decisions[name_to_index[parameters["expert"]]],
        parameters, corrections,
    )
    fixed_evidence = evidence_sources[es]
    fixed_risk, fixed_phrase, fixed_task, _ = _metric(
        truth, fixed_prediction, fixed_evidence, np.arange(len(records))
    )
    adopted = bool(new_task >= base_task + .003
                   and fixed_task >= base_task + .002
                   and bootstrap["positive_fraction"] >= .80)
    payload = {
        "training_version": TRAINING_VERSION,
        "evaluation": "nested user-disjoint recalibration on all 1305 OOF posts",
        "baseline_recalibrated": {"risk_f1": base_risk,
                                  "phrase_f1": base_phrase_f1, "task1": base_task},
        "candidate_recalibrated": {"risk_f1": new_risk,
                                   "phrase_f1": new_phrase_f1, "task1": new_task,
                                   "folds": fold_rows},
        "fixed_production_diagnostic": {
            "hierarchy_mix": hm, "new_model_weight": nw, "evidence_source": es,
            **parameters, "risk_f1": fixed_risk,
            "phrase_f1": fixed_phrase, "task1": fixed_task,
        },
        "user_cluster_bootstrap": bootstrap,
        "adopted": adopted,
    }
    RESULTS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    CALIBRATION.write_text(json.dumps({
        "training_version": TRAINING_VERSION, "adopted": adopted,
        "hierarchy_mix": hm, "new_model_weight": nw, "evidence_source": es,
        **parameters, "baseline_task1": base_task, "candidate_task1": new_task,
        "bootstrap": bootstrap,
    }, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    return payload


if __name__ == "__main__":
    main()
