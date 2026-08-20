"""Nested user-disjoint Task-1 ablation for emoji and temporal semantics."""
from __future__ import annotations

import json
from collections import Counter

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC

from baseline import _post_phrase_f1
from configs.config import config
from inference.task1_evidence_v4 import apply_evidence_policy, decode_model_evidence
from inference.task1_polarity_v63 import polarity_candidate
from preprocess.task1_semantic_text import augment_task1_text, emoji_markers, temporal_markers
from trainer.task1_atomic_v25 import _load_records
from utils.task1_metric import task1_score


OUTPUT = config.OUTPUT_DIR / "task1_semantic_text_v71"
RESULTS = OUTPUT / "results.json"
TRAINING_VERSION = "task1-emoji-temporal-semantics-v71"


def _vectorizer():
    return FeatureUnion([
        ("word", TfidfVectorizer(lowercase=True, strip_accents="unicode",
                                  ngram_range=(1, 3), min_df=1,
                                  max_features=120_000, sublinear_tf=True)),
        ("char", TfidfVectorizer(lowercase=True, strip_accents="unicode",
                                  analyzer="char_wb", ngram_range=(3, 5),
                                  min_df=2, max_features=150_000,
                                  sublinear_tf=True)),
    ])


def _fit_predict(train_text, train_y, valid_text, c_value, balanced):
    vectorizer = _vectorizer()
    train_x = vectorizer.fit_transform(train_text)
    valid_x = vectorizer.transform(valid_text)
    model = LinearSVC(C=float(c_value),
                      class_weight="balanced" if balanced else None)
    model.fit(train_x, train_y)
    return model.decision_function(valid_x)


def _matrices(train_text, valid_text):
    vectorizer = _vectorizer()
    train_x = vectorizer.fit_transform(train_text)
    return train_x, vectorizer.transform(valid_text)


def _decision(train_x, train_y, valid_x, c_value, balanced):
    model = LinearSVC(C=float(c_value),
                      class_weight="balanced" if balanced else None)
    model.fit(train_x, train_y)
    return model.decision_function(valid_x)


def _softmax(values, temperature=1.0):
    values = np.asarray(values, dtype=np.float64) / float(temperature)
    values -= values.max(axis=1, keepdims=True)
    exp = np.exp(values)
    return exp / exp.sum(axis=1, keepdims=True)


def _decode_evidence(record, risk, decoder):
    spans = decode_model_evidence(
        record["text"], record["offsets"], record["start"], record["end"],
        threshold=float(decoder["threshold"]),
        max_tokens=int(decoder["max_tokens"]),
        end_policy=str(decoder["end_policy"]), limit=5,
    )
    return apply_evidence_policy(
        record["text"], int(risk), spans,
        policy=str(decoder["cue_policy"]), topk=int(decoder["topk"]),
    )


def _apply_polarity(records, prediction, evidence):
    prediction = prediction.copy(); evidence = list(evidence)
    for index, record in enumerate(records):
        candidate = polarity_candidate(record["text"], int(prediction[index]))
        if candidate is not None:
            prediction[index], evidence[index] = candidate
    return prediction, evidence


def _metric(truth, prediction, evidence, records, indices):
    indices = np.asarray(indices, dtype=np.int64)
    risk = float(f1_score(truth[indices], prediction[indices],
                          average="weighted", zero_division=0))
    phrase = float(np.mean([
        _post_phrase_f1(evidence[i], records[i]["gold"]) for i in indices
    ]))
    return risk, phrase, task1_score(risk, phrase)


def _candidate(base, probabilities, threshold, allowed):
    result = base.copy()
    confidence = probabilities.max(1)
    proposed = probabilities.argmax(1)
    change = allowed & (proposed != base) & (confidence >= float(threshold))
    result[change] = proposed[change]
    return result


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    records, membership_map, _ = _load_records()
    truth = np.asarray([int(row["truth"]) for row in records])
    global_indices = np.asarray([int(row["global_index"]) for row in records])
    membership = np.asarray([membership_map[int(i)] for i in global_indices])
    groups = np.asarray([str(row["user"]) for row in records])
    texts = np.asarray([str(row["text"]) for row in records], dtype=object)
    base = np.asarray([int(row["risk"]) for row in records])
    decoder = json.loads((config.OUTPUT_DIR / "task1_evidence_v4" / "calibration.json")
                         .read_text(encoding="utf-8"))
    base_evidence = [_decode_evidence(row, risk, decoder)
                     for row, risk in zip(records, base)]
    base, base_evidence = _apply_polarity(records, base, base_evidence)
    baseline = _metric(truth, base, base_evidence, records, np.arange(len(records)))

    modes = {
        "original": texts,
        "emoji": np.asarray([augment_task1_text(x, emoji=True, temporal=False)
                              for x in texts], dtype=object),
        "temporal": np.asarray([augment_task1_text(x, emoji=False, temporal=True)
                                 for x in texts], dtype=object),
        "emoji_temporal": np.asarray([augment_task1_text(x, emoji=True, temporal=True)
                                       for x in texts], dtype=object),
    }
    has_emoji = np.asarray([bool(emoji_markers(x)) for x in texts])
    has_temporal = np.asarray([bool(temporal_markers(x)) for x in texts])
    allowed_by_mode = {
        "original": np.ones(len(texts), dtype=bool),
        "emoji": has_emoji,
        "temporal": has_temporal,
        "emoji_temporal": has_emoji | has_temporal,
    }
    c_values = (.1, .25, .5, 1.0)
    thresholds = (.40, .50, .60, .70, .80)
    temperatures = (.7, 1.0, 1.3)
    crossfit = base.copy(); crossfit_evidence = list(base_evidence)
    fold_rows = []; selections = []
    for outer_fold in range(4):
        fit = np.flatnonzero(membership != outer_fold)
        held = np.flatnonzero(membership == outer_fold)
        inner_folds = [fold for fold in range(4) if fold != outer_fold]
        candidates = []
        for mode, mode_text in modes.items():
            decision_cache = {
                (c_value, balanced): np.zeros((len(texts), 4), dtype=np.float32)
                for c_value in c_values for balanced in (False, True)
            }
            for inner_fold in inner_folds:
                inner_train = np.flatnonzero(
                    (membership != outer_fold) & (membership != inner_fold))
                inner_valid = np.flatnonzero(membership == inner_fold)
                train_x, valid_x = _matrices(
                    mode_text[inner_train], mode_text[inner_valid])
                for c_value in c_values:
                    for balanced in (False, True):
                        decision_cache[(c_value, balanced)][inner_valid] = _decision(
                            train_x, truth[inner_train], valid_x, c_value, balanced)
            for c_value in c_values:
                for balanced in (False, True):
                    inner_decision = decision_cache[(c_value, balanced)]
                    for temperature in temperatures:
                        probability = _softmax(inner_decision, temperature)
                        for threshold in thresholds:
                            prediction = _candidate(
                                base, probability, threshold, allowed_by_mode[mode])
                            evidence = [
                                base_evidence[i] if prediction[i] == base[i]
                                else _decode_evidence(records[i], prediction[i], decoder)
                                for i in range(len(records))
                            ]
                            prediction, evidence = _apply_polarity(
                                records, prediction, evidence)
                            score = _metric(truth, prediction, evidence, records, fit)
                            candidates.append((score[2], score[0], mode, c_value,
                                               balanced, temperature, threshold))
        selected = max(candidates, key=lambda row: (row[0], row[1]))
        _, _, mode, c_value, balanced, temperature, threshold = selected
        decision = _fit_predict(modes[mode][fit], truth[fit], modes[mode][held],
                                c_value, balanced)
        probability = _softmax(decision, temperature)
        local = _candidate(base[held], probability, threshold,
                           allowed_by_mode[mode][held])
        evidence = [
            base_evidence[i] if local[j] == base[i]
            else _decode_evidence(records[i], local[j], decoder)
            for j, i in enumerate(held)
        ]
        local, evidence = _apply_polarity([records[i] for i in held], local, evidence)
        crossfit[held] = local
        for j, i in enumerate(held):
            crossfit_evidence[i] = evidence[j]
        old_score = _metric(truth, base, base_evidence, records, held)
        new_score = _metric(truth, crossfit, crossfit_evidence, records, held)
        row = {"fold": outer_fold, "mode": mode, "c": c_value,
               "balanced": balanced, "temperature": temperature,
               "confidence_threshold": threshold,
               "changed": int((crossfit[held] != base[held]).sum()),
               "baseline_task1": old_score[2], "candidate_task1": new_score[2]}
        fold_rows.append(row); selections.append(row)
        print(f"V71 fold={outer_fold} mode={mode} changed={row['changed']} "
              f"task1 {old_score[2]:.6f}->{new_score[2]:.6f}", flush=True)

    candidate = _metric(truth, crossfit, crossfit_evidence,
                        records, np.arange(len(records)))
    unique = np.unique(groups); rng = np.random.default_rng(config.SEED + 7171)
    deltas = []
    for _ in range(4000):
        sampled = rng.choice(unique, len(unique), replace=True)
        indices = np.concatenate([np.flatnonzero(groups == user) for user in sampled])
        deltas.append(_metric(truth, crossfit, crossfit_evidence, records, indices)[2]
                      - _metric(truth, base, base_evidence, records, indices)[2])
    deltas = np.asarray(deltas)
    chosen_modes = Counter(row["mode"] for row in selections)
    adopted = bool(candidate[2] >= baseline[2] + .002
                   and float((deltas > 0).mean()) >= .80
                   and sum(row["candidate_task1"] >= row["baseline_task1"]
                           for row in fold_rows) >= 3)
    payload = {
        "training_version": TRAINING_VERSION,
        "evaluation": "four-fold nested user-disjoint selective semantic expert",
        "coverage": {"emoji_or_emoticon_posts": int(has_emoji.sum()),
                     "temporal_scope_posts": int(has_temporal.sum())},
        "baseline": {"risk_f1": baseline[0], "phrase_f1": baseline[1],
                     "task1": baseline[2]},
        "candidate": {"risk_f1": candidate[0], "phrase_f1": candidate[1],
                      "task1": candidate[2], "folds": fold_rows,
                      "selected_mode_counts": dict(chosen_modes)},
        "bootstrap": {"mean_delta": float(deltas.mean()),
                      "p05_delta": float(np.quantile(deltas, .05)),
                      "p95_delta": float(np.quantile(deltas, .95)),
                      "positive_fraction": float((deltas > 0).mean())},
        "adopted": adopted,
    }
    RESULTS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    return payload


if __name__ == "__main__":
    main()
