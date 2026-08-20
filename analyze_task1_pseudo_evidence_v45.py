"""OOF pseudo-evidence reweighting for the Task 1 lexical risk expert (V45).

The experiment follows the data-augmentation theme of successful earlier
BigData Cup systems, but does not use gold evidence at inference or validation.
Every post receives a pseudo-rationale from a user-disjoint evidence model.  A
second lexical expert sees that rationale repeated next to the untouched post,
which changes TF-IDF weights without breaking the original evidence offsets.
"""
from __future__ import annotations

from collections import Counter
import json

import joblib
import numpy as np
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC

from analyze_task1_lexical_v11 import _softmax
from analyze_task1_oof_risk_v36 import (
    CACHE as V36_CACHE, _evidence_matrix,
)
from baseline import _vectorizer
from configs.config import config
from inference.task1_evidence_v4 import (
    apply_evidence_policy, correct_risk_only, decode_model_evidence,
)
from preprocess.preprocess import load_train_data
from trainer.task1_atomic_v25 import _load_records
from utils.task1_metric import task1_score


OUTPUT = config.OUTPUT_DIR / "task1_pseudo_evidence_v45"
RESULTS = OUTPUT / "results.json"
CALIBRATION = OUTPUT / "calibration.json"
ARTIFACT = OUTPUT / "full_pseudo_evidence_svc.joblib"
TRAINING_VERSION = "task1-oof-pseudo-evidence-lexical-v45"


def _recipes():
    path = config.OUTPUT_DIR / "task1_oof_decoder_v35" / "calibration.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload.get("adopted", False):
        raise RuntimeError("V45 requires the adopted V35 evidence decoder")
    return payload["parameters_by_predicted_risk"]


def pseudo_evidence(record, recipes):
    """Decode evidence without consulting the post's gold risk or gold spans."""
    risk = int(np.argmax(np.asarray(record["old_probability"])))
    risk = correct_risk_only(record["text"], risk)
    parameters = recipes[config.ID2RISK[risk]]
    spans = decode_model_evidence(
        record["text"], record["offsets"], record["start"], record["end"],
        threshold=float(parameters["threshold"]),
        max_tokens=int(parameters["max_tokens"]),
        end_policy=str(parameters["end_policy"]), limit=5,
    )
    return apply_evidence_policy(
        record["text"], risk, spans,
        policy=str(parameters["cue_policy"]), topk=int(parameters["topk"]),
    )


def evidence_view(record, recipes, mode):
    text = str(record["text"])
    evidence = " ; ".join(pseudo_evidence(record, recipes)).strip()
    if not evidence or mode == "raw":
        return text
    if mode == "prefix2":
        return f"{evidence} {evidence} {text}"
    if mode == "suffix2":
        return f"{text} {evidence} {evidence}"
    if mode == "evidence_only":
        return evidence
    raise ValueError(f"Unknown pseudo-evidence view: {mode}")


def _oof_decisions(records, membership, truth, recipes):
    modes = ("prefix2", "suffix2", "evidence_only")
    specifications = [
        (mode, c_value, balanced)
        for mode in modes
        for c_value in (.25, .5, 1.0)
        for balanced in (False, True)
    ]
    names = [
        f"{mode}-c{c_value:g}-{'balanced' if balanced else 'plain'}"
        for mode, c_value, balanced in specifications
    ]
    decisions = np.zeros((len(specifications), len(records), 4), dtype=np.float32)
    texts_by_mode = {
        mode: [evidence_view(record, recipes, mode) for record in records]
        for mode in modes
    }
    for fold in range(4):
        fit = np.flatnonzero(membership != fold)
        held = np.flatnonzero(membership == fold)
        for mode in modes:
            vectorizer = _vectorizer()
            train_matrix = vectorizer.fit_transform([texts_by_mode[mode][i] for i in fit])
            held_matrix = vectorizer.transform([texts_by_mode[mode][i] for i in held])
            for expert_index, (candidate_mode, c_value, balanced) in enumerate(specifications):
                if candidate_mode != mode:
                    continue
                model = LinearSVC(
                    C=float(c_value), class_weight="balanced" if balanced else None,
                ).fit(train_matrix, truth[fit])
                decisions[expert_index, held] = model.decision_function(held_matrix)
        print(f"V45 pseudo-evidence lexical OOF fold {fold + 1}/4", flush=True)
    return names, decisions, texts_by_mode


def _biased_base_probability(transformer, raw_decision, parameters):
    lexical = _softmax(raw_decision, parameters["temperature"])
    probability = ((1.0 - parameters["lexical_weight"]) * transformer
                   + parameters["lexical_weight"] * lexical)
    logits = np.log(np.clip(probability, 1e-8, 1.0))
    logits[:, 0] += float(parameters.get("indicator_bias", 0.0))
    logits[:, 2] += float(parameters.get("behavior_bias", 0.0))
    logits[:, 3] += float(parameters.get("attempt_bias", 0.0))
    logits -= logits.max(1, keepdims=True)
    probability = np.exp(logits)
    return probability / probability.sum(1, keepdims=True)


def _predict(base_probability, decision, temperature, weight, texts):
    candidate = _softmax(decision, temperature)
    probability = (1.0 - weight) * base_probability + weight * candidate
    raw = probability.argmax(1)
    return np.asarray([
        correct_risk_only(text, int(risk)) for text, risk in zip(texts, raw)
    ], dtype=np.int64)


def _metric(truth, prediction, evidence, indices):
    indices = np.asarray(indices, dtype=np.int64)
    risk = float(f1_score(
        truth[indices], prediction[indices], average="weighted", zero_division=0,
    ))
    phrase_values = evidence[np.arange(len(prediction)), prediction]
    phrase = float(phrase_values[indices].mean())
    return risk, phrase, task1_score(risk, phrase), phrase_values


def _select(truth, base_probability, names, decisions, texts, evidence, fit):
    rows = []
    for expert_index, name in enumerate(names):
        for temperature in (.5, .75, 1.0):
            for weight in (0.0, .1, .2, .3, .4):
                if weight == 0.0 and (expert_index or temperature != .5):
                    continue
                prediction = _predict(
                    base_probability, decisions[expert_index], temperature, weight, texts,
                )
                metric = _metric(truth, prediction, evidence, fit)
                rows.append({"expert": name, "temperature": temperature,
                             "pseudo_evidence_weight": weight,
                             "risk_f1": metric[0], "phrase_f1": metric[1],
                             "task1": metric[2], "prediction": prediction})
    return max(rows, key=lambda row: (row["task1"], row["risk_f1"]))


def _aggregate(rows):
    def mode(key):
        return Counter(row[key] for row in rows).most_common(1)[0][0]
    return {"expert": mode("expert"), "temperature": float(mode("temperature")),
            "pseudo_evidence_weight": float(mode("pseudo_evidence_weight"))}


def _bootstrap(truth, groups, baseline, candidate, base_phrase, new_phrase):
    unique = np.unique(groups); rng = np.random.default_rng(config.SEED + 4545)
    deltas = []
    for _ in range(4000):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        positions = np.concatenate([np.flatnonzero(groups == user) for user in sampled])
        old_risk = f1_score(truth[positions], baseline[positions],
                            average="weighted", zero_division=0)
        new_risk = f1_score(truth[positions], candidate[positions],
                            average="weighted", zero_division=0)
        deltas.append(task1_score(old_risk, float(base_phrase[positions].mean())) * -1
                      + task1_score(new_risk, float(new_phrase[positions].mean())))
    values = np.asarray(deltas)
    return {"mean_task1_delta": float(values.mean()),
            "p05_task1_delta": float(np.quantile(values, .05)),
            "p95_task1_delta": float(np.quantile(values, .95)),
            "positive_fraction": float((values > 0).mean())}


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame = load_train_data().reset_index(drop=True)
    labels = frame.risk_label.to_numpy(dtype=np.int64)
    groups = frame.anon_user_id.astype(str).to_numpy()
    records, membership_map, outer_raw = _load_records()
    global_indices = np.asarray([int(row["global_index"]) for row in records])
    membership = np.asarray([membership_map[int(index)] for index in global_indices])
    truth = labels[global_indices]
    texts = [str(row["text"]) for row in records]
    local_groups = groups[global_indices]
    recipes = _recipes()

    names, decisions, _ = _oof_decisions(records, membership, truth, recipes)
    saved = np.load(V36_CACHE, allow_pickle=True)
    old_names = saved["names"].tolist(); old_decisions = saved["decisions"]
    v36 = json.loads((config.OUTPUT_DIR / "task1_oof_risk_v36" / "calibration.json")
                     .read_text(encoding="utf-8"))
    transformer = np.vstack([row["old_probability"] for row in records])
    base_probability = _biased_base_probability(
        transformer, old_decisions[old_names.index(v36["expert"])], v36,
    )
    evidence = _evidence_matrix(records)
    baseline = _predict(base_probability, decisions[0], .5, 0.0, texts)
    base_metric = _metric(truth, baseline, evidence, np.arange(len(truth)))

    crossfit = np.zeros(len(truth), dtype=np.int64); selections = []; folds = []
    for fold in range(4):
        fit = np.flatnonzero(membership != fold)
        held = np.flatnonzero(membership == fold)
        selected = _select(
            truth, base_probability, names, decisions, texts, evidence, fit,
        )
        crossfit[held] = selected["prediction"][held]
        old = _metric(truth, baseline, evidence, held)
        new = _metric(truth, crossfit, evidence, held)
        compact = {key: selected[key] for key in (
            "expert", "temperature", "pseudo_evidence_weight",
        )}
        selections.append(compact)
        folds.append({"fold": fold, "posts": int(len(held)), **compact,
                      "changed_predictions": int(np.sum(
                          baseline[held] != crossfit[held])),
                      "baseline_task1": old[2], "candidate_task1": new[2]})
        print(f"V45 fold={fold} {compact['expert']} weight="
              f"{compact['pseudo_evidence_weight']:.2f} "
              f"task1 {old[2]:.6f}->{new[2]:.6f}", flush=True)

    cross_metric = _metric(truth, crossfit, evidence, np.arange(len(truth)))
    production = _aggregate(selections)
    expert_index = names.index(production["expert"])
    fixed = _predict(base_probability, decisions[expert_index],
                     production["temperature"],
                     production["pseudo_evidence_weight"], texts)
    fixed_metric = _metric(truth, fixed, evidence, np.arange(len(truth)))
    bootstrap = _bootstrap(truth, local_groups, baseline, crossfit,
                           base_metric[3], cross_metric[3])
    adopted = bool(cross_metric[2] >= base_metric[2] + .003
                   and fixed_metric[2] >= base_metric[2] + .003
                   and bootstrap["positive_fraction"] >= .80)

    if adopted:
        all_records = list(records) + list(outer_raw["records"])
        by_index = {int(row["global_index"]): row for row in all_records}
        ordered = [by_index[index] for index in range(len(frame))]
        mode = production["expert"].split("-c", 1)[0]
        views = [evidence_view(record, recipes, mode) for record in ordered]
        vectorizer = _vectorizer(); matrix = vectorizer.fit_transform(views)
        tail = production["expert"].split("-c", 1)[1]
        c_value = float(tail.rsplit("-", 1)[0])
        balanced = production["expert"].endswith("-balanced")
        model = LinearSVC(C=c_value,
                          class_weight="balanced" if balanced else None)
        model.fit(matrix, labels)
        joblib.dump({"training_version": TRAINING_VERSION, **production,
                     "mode": mode, "vectorizer": vectorizer, "model": model}, ARTIFACT)

    payload = {"training_version": TRAINING_VERSION,
        "method": "OOF evidence decoder -> pseudo-rationale TF-IDF view -> LinearSVC",
        "gold_evidence_used_as_model_input": False,
        "baseline_v36": {"risk_f1": base_metric[0], "phrase_f1": base_metric[1],
                         "task1": base_metric[2]},
        "crossfit_candidate": {"risk_f1": cross_metric[0],
            "phrase_f1": cross_metric[1], "task1": cross_metric[2], "folds": folds},
        "fixed_production_diagnostic": {**production,
            "risk_f1": fixed_metric[0], "phrase_f1": fixed_metric[1],
            "task1": fixed_metric[2]},
        "user_cluster_bootstrap": bootstrap, "adopted": adopted}
    RESULTS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    CALIBRATION.write_text(json.dumps({"training_version": TRAINING_VERSION,
        "adopted": adopted, **production, "crossfit_task1": cross_metric[2]},
        indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    return payload


if __name__ == "__main__":
    main()
