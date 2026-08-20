"""Evidence-conditioned hierarchical Task-1 risk classifier (V75).

Inspired by 2025 work that decomposes ordinal suicide risk and first isolates
evidence-bearing sentences.  The hierarchy factorises the four-way decision:

    explicit suicide? -> past/recent attempt? -> plan/behaviour or ideation?

All specialist probabilities are out-of-fold by user.  Hyper-parameters are
selected in a nested fashion inside the 1,305 outer-training posts, and the
resulting fixed recipe is evaluated once on 330 untouched users' posts.
"""
from __future__ import annotations

from collections import Counter
import json
import re

import joblib
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.pipeline import FeatureUnion

from analyze_task1_lexical_v11 import _lexical_experts, _softmax
from analyze_task1_oof_risk_v36 import CACHE as V36_CACHE
from baseline import _post_phrase_f1
from configs.config import config
from inference.task1_evidence_v4 import (
    apply_evidence_policy, correct_risk_only, decode_model_evidence,
)
from inference.task1_polarity_v63 import polarity_candidate
from preprocess.preprocess import load_train_data
from trainer.task1_atomic_v25 import _load_records, _segment_spans
from utils.task1_metric import task1_score


OUTPUT = config.OUTPUT_DIR / "task1_hierarchical_evidence_v75"
RESULTS = OUTPUT / "results.json"
MODEL = OUTPUT / "model.joblib"
TRAINING_VERSION = "task1-evidence-conditioned-hierarchy-v75"


def _vectorizer():
    return FeatureUnion([
        ("word", TfidfVectorizer(lowercase=True, strip_accents="unicode",
                                  ngram_range=(1, 3), min_df=1,
                                  max_features=130_000, sublinear_tf=True)),
        ("char", TfidfVectorizer(lowercase=True, strip_accents="unicode",
                                  analyzer="char_wb", ngram_range=(3, 5),
                                  min_df=2, max_features=150_000,
                                  sublinear_tf=True)),
    ])


def _raw_evidence(record):
    """Return evidence-bearing sentence context without using a risk label."""
    text = str(record["text"])
    spans = decode_model_evidence(
        text, record["offsets"], record["start"], record["end"],
        threshold=.30, max_tokens=16, end_policy="best", limit=8,
    )
    selected = []
    segments = _segment_spans(text)
    for phrase in spans:
        start = text.casefold().find(str(phrase).casefold())
        if start < 0:
            continue
        end = start + len(str(phrase))
        for left, right in segments:
            if left < end and right > start:
                selected.append(text[left:right].strip())
                break
    # Preserve short model spans even when sentence localisation failed.
    selected.extend(str(span).strip() for span in spans)
    return " ".join(list(dict.fromkeys(x for x in selected if x))[:6])


def _documents(texts, snippets, mode):
    if mode == "full":
        return np.asarray(texts, dtype=object)
    if mode == "evidence":
        return np.asarray([snippet or text for text, snippet in zip(texts, snippets)],
                          dtype=object)
    if mode == "full_plus_evidence":
        return np.asarray([
            f"{text}\n[EVIDENCE] {snippet}\n[EVIDENCE] {snippet}"
            if snippet else str(text)
            for text, snippet in zip(texts, snippets)
        ], dtype=object)
    raise ValueError(mode)


def _binary_probability(matrix_train, target, matrix_valid, c_value, balanced):
    target = np.asarray(target, dtype=np.int64)
    if len(np.unique(target)) < 2:
        return np.full(matrix_valid.shape[0], float(target[0]), dtype=np.float32), None
    model = LogisticRegression(
        C=float(c_value), class_weight="balanced" if balanced else None,
        solver="liblinear", max_iter=1500,
    ).fit(matrix_train, target)
    return model.predict_proba(matrix_valid)[:, list(model.classes_).index(1)], model


def _fit_hierarchy(train_documents, train_labels, valid_documents,
                   c_value, balanced, return_model=False):
    vectorizer = _vectorizer()
    train_x = vectorizer.fit_transform(train_documents)
    valid_x = vectorizer.transform(valid_documents)
    labels = np.asarray(train_labels, dtype=np.int64)
    explicit, explicit_model = _binary_probability(
        train_x, labels > 0, valid_x, c_value, balanced)
    suicidal = labels > 0
    attempt, attempt_model = _binary_probability(
        train_x[suicidal], labels[suicidal] == 3,
        valid_x, c_value, balanced)
    ideation_behavior = (labels == 1) | (labels == 2)
    behavior, behavior_model = _binary_probability(
        train_x[ideation_behavior], labels[ideation_behavior] == 2,
        valid_x, c_value, balanced)
    probability = np.column_stack((
        1.0 - explicit,
        explicit * (1.0 - attempt) * (1.0 - behavior),
        explicit * (1.0 - attempt) * behavior,
        explicit * attempt,
    )).astype(np.float32)
    probability /= probability.sum(1, keepdims=True).clip(1e-8)
    bundle = None
    if return_model:
        bundle = {"vectorizer": vectorizer, "explicit": explicit_model,
                  "attempt": attempt_model, "behavior": behavior_model}
    return probability, bundle


def _base_probability(transformer, lexical_decision, parameters):
    lexical = _softmax(lexical_decision, float(parameters["temperature"]))
    probability = ((1.0 - float(parameters["lexical_weight"])) * transformer
                   + float(parameters["lexical_weight"]) * lexical)
    logits = np.log(np.clip(probability, 1e-8, 1.0))
    logits[:, 0] += float(parameters.get("indicator_bias", 0.0))
    logits[:, 2] += float(parameters.get("behavior_bias", 0.0))
    logits[:, 3] += float(parameters.get("attempt_bias", 0.0))
    logits -= logits.max(1, keepdims=True)
    probability = np.exp(logits)
    return probability / probability.sum(1, keepdims=True)


def _lookup(records, decoder):
    risk = np.zeros((len(records), 4), dtype=np.int64)
    phrase = np.zeros((len(records), 4), dtype=np.float32)
    for i, record in enumerate(records):
        for raw in range(4):
            corrected = correct_risk_only(record["text"], raw)
            spans = decode_model_evidence(
                record["text"], record["offsets"], record["start"], record["end"],
                threshold=float(decoder["threshold"]),
                max_tokens=int(decoder["max_tokens"]),
                end_policy=str(decoder["end_policy"]), limit=5)
            evidence = apply_evidence_policy(
                record["text"], corrected, spans,
                policy=str(decoder["cue_policy"]), topk=int(decoder["topk"]))
            polarity = polarity_candidate(record["text"], corrected)
            if polarity is not None:
                corrected, evidence = polarity
            risk[i, raw] = corrected
            phrase[i, raw] = _post_phrase_f1(evidence, record["gold"])
    return risk, phrase


def _predict(base, hierarchy, weight, temperature, biases, risk_lookup):
    hierarchy = np.clip(hierarchy, 1e-7, 1.0) ** (1.0 / float(temperature))
    hierarchy /= hierarchy.sum(1, keepdims=True)
    probability = (1.0 - float(weight)) * base + float(weight) * hierarchy
    logits = np.log(np.clip(probability, 1e-8, 1.0)) + np.asarray(biases)[None, :]
    raw = logits.argmax(1)
    return raw, risk_lookup[np.arange(len(raw)), raw]


def _metric(truth, raw, prediction, phrase_lookup, indices):
    indices = np.asarray(indices, dtype=np.int64)
    risk = float(f1_score(truth[indices], prediction[indices],
                          average="weighted", zero_division=0))
    values = phrase_lookup[np.arange(len(raw)), raw]
    phrase = float(values[indices].mean())
    return risk, phrase, task1_score(risk, phrase), values


def _parameter_grid(truth, base, experts, risk_lookup, phrase_lookup, fit):
    rows = []
    for expert_name, hierarchy in experts.items():
        for weight in (0.0, .05, .10, .15, .20, .30, .40):
            if weight == 0 and expert_name != next(iter(experts)):
                continue
            for temperature in (.7, 1.0, 1.3):
                if weight == 0 and temperature != .7:
                    continue
                for behavior_bias in (-.10, 0.0, .10):
                    for attempt_bias in (-.10, 0.0, .10):
                        biases = (0.0, 0.0, behavior_bias, attempt_bias)
                        raw, prediction = _predict(
                            base, hierarchy, weight, temperature, biases, risk_lookup)
                        risk, phrase, score, _ = _metric(
                            truth, raw, prediction, phrase_lookup, fit)
                        rows.append({"expert": expert_name, "weight": weight,
                                     "temperature": temperature,
                                     "behavior_bias": behavior_bias,
                                     "attempt_bias": attempt_bias,
                                     "risk_f1": risk, "phrase_f1": phrase,
                                     "task1": score})
    return max(rows, key=lambda row: (row["task1"], row["risk_f1"], -row["weight"]))


def _aggregate(rows):
    def mode(key): return Counter(row[key] for row in rows).most_common(1)[0][0]
    return {key: mode(key) for key in (
        "expert", "weight", "temperature", "behavior_bias", "attempt_bias")}


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame = load_train_data().reset_index(drop=True)
    labels = frame.risk_label.to_numpy(dtype=np.int64)
    groups_all = frame.anon_user_id.astype(str).to_numpy()
    records, membership_map, outer_raw = _load_records()
    global_indices = np.asarray([int(row["global_index"]) for row in records])
    truth = labels[global_indices]
    membership = np.asarray([membership_map[int(i)] for i in global_indices])
    groups = groups_all[global_indices]
    texts = np.asarray([str(row["text"]) for row in records], dtype=object)
    snippets = np.asarray([_raw_evidence(row) for row in records], dtype=object)

    v36 = json.loads((config.OUTPUT_DIR / "task1_oof_risk_v36" / "calibration.json")
                     .read_text(encoding="utf-8"))
    saved = np.load(V36_CACHE, allow_pickle=True)
    names = saved["names"].tolist()
    lexical = saved["decisions"][names.index(v36["expert"])]
    transformer = np.vstack([row["old_probability"] for row in records])
    base = _base_probability(transformer, lexical, v36)
    decoder = json.loads((config.OUTPUT_DIR / "task1_evidence_v4" / "calibration.json")
                         .read_text(encoding="utf-8"))
    risk_lookup, phrase_lookup = _lookup(records, decoder)
    base_raw = base.argmax(1)
    base_prediction = risk_lookup[np.arange(len(base_raw)), base_raw]
    baseline = _metric(truth, base_raw, base_prediction, phrase_lookup,
                       np.arange(len(truth)))

    modes = ("full", "evidence", "full_plus_evidence")
    c_values = (.10, .25, .50, 1.0)
    experts = {}
    for mode in modes:
        documents = _documents(texts, snippets, mode)
        for c_value in c_values:
            for balanced in (False, True):
                key = f"{mode}|c={c_value}|balanced={int(balanced)}"
                probability = np.zeros((len(records), 4), dtype=np.float32)
                for fold in range(4):
                    fit = np.flatnonzero(membership != fold)
                    held = np.flatnonzero(membership == fold)
                    probability[held], _ = _fit_hierarchy(
                        documents[fit], truth[fit], documents[held],
                        c_value, balanced)
                experts[key] = probability
                print(f"V75 built OOF hierarchy {key}", flush=True)

    cross_raw = base_raw.copy(); cross_prediction = base_prediction.copy()
    cross_phrase_values = phrase_lookup[np.arange(len(base_raw)), base_raw].copy()
    selections = []; fold_rows = []
    for fold in range(4):
        fit = np.flatnonzero(membership != fold); held = np.flatnonzero(membership == fold)
        selected = _parameter_grid(
            truth, base, experts, risk_lookup, phrase_lookup, fit)
        hierarchy = experts[selected["expert"]]
        biases = (0.0, 0.0, selected["behavior_bias"], selected["attempt_bias"])
        raw, prediction = _predict(base, hierarchy, selected["weight"],
                                   selected["temperature"], biases, risk_lookup)
        cross_raw[held] = raw[held]; cross_prediction[held] = prediction[held]
        cross_phrase_values[held] = phrase_lookup[held, raw[held]]
        old = _metric(truth, base_raw, base_prediction, phrase_lookup, held)
        new_risk = float(f1_score(truth[held], cross_prediction[held],
                                  average="weighted", zero_division=0))
        new_phrase = float(cross_phrase_values[held].mean())
        new_task = task1_score(new_risk, new_phrase)
        fold_rows.append({"fold": fold, **{k: selected[k] for k in (
            "expert", "weight", "temperature", "behavior_bias", "attempt_bias")},
            "changed": int((cross_prediction[held] != base_prediction[held]).sum()),
            "baseline_task1": old[2], "candidate_task1": new_task})
        selections.append(selected)
        print(f"V75 fold={fold} weight={selected['weight']} "
              f"task1 {old[2]:.6f}->{new_task:.6f}", flush=True)

    cross_risk = float(f1_score(truth, cross_prediction,
                                average="weighted", zero_division=0))
    cross_phrase = float(cross_phrase_values.mean())
    cross_task = task1_score(cross_risk, cross_phrase)
    production = _aggregate(selections)
    hierarchy = experts[production["expert"]]
    biases = (0.0, 0.0, production["behavior_bias"], production["attempt_bias"])
    fixed_raw, fixed_prediction = _predict(
        base, hierarchy, production["weight"], production["temperature"],
        biases, risk_lookup)
    fixed = _metric(truth, fixed_raw, fixed_prediction, phrase_lookup,
                    np.arange(len(truth)))

    # Fit the fixed expert on all outer-training users, then evaluate only once
    # on the untouched 330-post outer user split.
    outer_train = global_indices
    outer_valid = np.asarray(outer_raw["valid_idx"], dtype=np.int64)
    outer_records = outer_raw["records"]
    all_record = {int(row["global_index"]): row for row in records}
    all_record.update({int(index): row for index, row in zip(outer_valid, outer_records)})
    all_snippets = np.asarray([_raw_evidence(all_record[i]) for i in range(len(frame))],
                              dtype=object)
    mode_match = re.match(r"(.+)\|c=([0-9.]+)\|balanced=([01])", production["expert"])
    mode, c_value, balanced = mode_match.group(1), float(mode_match.group(2)), bool(int(mode_match.group(3)))
    all_documents = _documents(frame.text.astype(str).to_numpy(), all_snippets, mode)
    outer_hierarchy, bundle = _fit_hierarchy(
        all_documents[outer_train], labels[outer_train], all_documents[outer_valid],
        c_value, balanced, return_model=True)
    outer_lexical = _lexical_experts(frame, outer_train, outer_valid)[v36["expert"]]
    outer_transformer = np.vstack([row["old_probability"] for row in outer_records])
    outer_base = _base_probability(outer_transformer, outer_lexical, v36)
    outer_risk_lookup, outer_phrase_lookup = _lookup(outer_records, decoder)
    outer_base_raw = outer_base.argmax(1)
    outer_base_prediction = outer_risk_lookup[np.arange(len(outer_records)), outer_base_raw]
    outer_baseline = _metric(labels[outer_valid], outer_base_raw,
                             outer_base_prediction, outer_phrase_lookup,
                             np.arange(len(outer_valid)))
    outer_raw_prediction, outer_prediction = _predict(
        outer_base, outer_hierarchy, production["weight"], production["temperature"],
        biases, outer_risk_lookup)
    outer_candidate = _metric(labels[outer_valid], outer_raw_prediction,
                              outer_prediction, outer_phrase_lookup,
                              np.arange(len(outer_valid)))

    unique = np.unique(groups_all[outer_valid])
    rng = np.random.default_rng(config.SEED + 7575); deltas = []
    outer_base_phrase = outer_phrase_lookup[np.arange(len(outer_valid)), outer_base_raw]
    outer_new_phrase = outer_phrase_lookup[np.arange(len(outer_valid)), outer_raw_prediction]
    for _ in range(4000):
        sampled = rng.choice(unique, len(unique), replace=True)
        positions = np.concatenate([
            np.flatnonzero(groups_all[outer_valid] == user) for user in sampled])
        old_risk = f1_score(labels[outer_valid][positions], outer_base_prediction[positions],
                            average="weighted", zero_division=0)
        new_risk = f1_score(labels[outer_valid][positions], outer_prediction[positions],
                            average="weighted", zero_division=0)
        deltas.append(task1_score(new_risk, float(outer_new_phrase[positions].mean()))
                      - task1_score(old_risk, float(outer_base_phrase[positions].mean())))
    deltas = np.asarray(deltas)
    bootstrap = {"mean_delta": float(deltas.mean()),
                 "p05_delta": float(np.quantile(deltas, .05)),
                 "p95_delta": float(np.quantile(deltas, .95)),
                 "positive_fraction": float((deltas > 0).mean())}
    adopted = bool(outer_candidate[2] >= outer_baseline[2] + .002
                   and bootstrap["positive_fraction"] >= .80
                   and bootstrap["p05_delta"] >= 0
                   and cross_task >= baseline[2])
    if adopted:
        joblib.dump({"training_version": TRAINING_VERSION,
                     "parameters": production, "mode": mode,
                     "c": c_value, "balanced": balanced, **bundle}, MODEL)
    payload = {
        "training_version": TRAINING_VERSION,
        "method": "explicit -> attempt -> behavior/ideation evidence-conditioned hierarchy",
        "evaluation": "four-fold nested user OOF selection; untouched 330-post outer user holdout",
        "oof_baseline": {"risk_f1": baseline[0], "phrase_f1": baseline[1],
                         "task1": baseline[2]},
        "oof_crossfit": {"risk_f1": cross_risk, "phrase_f1": cross_phrase,
                         "task1": cross_task, "folds": fold_rows},
        "fixed_oof": {**production, "risk_f1": fixed[0],
                      "phrase_f1": fixed[1], "task1": fixed[2]},
        "outer_baseline": {"risk_f1": outer_baseline[0],
                           "phrase_f1": outer_baseline[1], "task1": outer_baseline[2]},
        "outer_candidate": {"risk_f1": outer_candidate[0],
                            "phrase_f1": outer_candidate[1], "task1": outer_candidate[2],
                            "changed": int((outer_prediction != outer_base_prediction).sum()),
                            "confusion": confusion_matrix(labels[outer_valid], outer_prediction,
                                                           labels=np.arange(4)).tolist()},
        "bootstrap": bootstrap, "adopted": adopted,
    }
    RESULTS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    return payload


if __name__ == "__main__":
    main()
