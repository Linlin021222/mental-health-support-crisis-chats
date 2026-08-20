"""Conservative boundary-only use of the V25 atomic evidence model."""
from __future__ import annotations

import json
import re

import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from transformers import AutoTokenizer

from baseline import _post_phrase_f1
from configs.config import config
from inference.task1_evidence_v4 import load_evidence_calibration
from preprocess.preprocess import load_train_data
from trainer.task1_atomic_v25 import (
    INNER_CALIBRATION_FOLD,
    INNER_CHECKPOINT,
    OUTER_CHECKPOINT,
    OUTPUT as V25_OUTPUT,
    AtomicEvidenceModel,
    _atomic_candidates,
    _baseline_evidence,
    _bootstrap,
    _build_examples,
    _infer,
    _load_records,
    _normalise,
)
from utils.task1_metric import task1_score


OUTPUT = config.OUTPUT_DIR / "task1_atomic_refine_v26"
RESULTS = OUTPUT / "results.json"
CALIBRATION = OUTPUT / "calibration.json"
RAW = OUTPUT / "atomic_outputs.pt"
TRAINING_VERSION = "task1-atomic-boundary-refine-v26"


def _occurrences(text, phrase):
    return [
        (match.start(), match.end())
        for match in re.finditer(re.escape(str(phrase)), text, flags=re.IGNORECASE)
    ]


def _refine_one(text, baseline, atomic, parameters):
    result = []
    for phrase in baseline:
        baseline_words = max(1, len(str(phrase).split()))
        if baseline_words < parameters["minimum_baseline_tokens"]:
            result.append(phrase); continue
        choices = []
        for score, atomic_start, atomic_end, atomic_phrase in atomic:
            if score < parameters["replacement_gate"]:
                continue
            atomic_normal = _normalise(atomic_phrase)
            baseline_normal = _normalise(phrase)
            if not atomic_normal or atomic_normal == baseline_normal:
                continue
            word_ratio = len(atomic_phrase.split()) / baseline_words
            if word_ratio < parameters["minimum_length_ratio"]:
                continue
            for baseline_start, baseline_end in _occurrences(text, phrase):
                intersection = max(
                    0, min(atomic_end, baseline_end) - max(atomic_start, baseline_start)
                )
                contained = atomic_start >= baseline_start and atomic_end <= baseline_end
                overlap = intersection / max(1, atomic_end - atomic_start)
                allowed = contained if parameters["alignment"] == "contained" else overlap >= 0.80
                if allowed and atomic_normal in baseline_normal:
                    choices.append((score, len(atomic_phrase), atomic_phrase))
        if choices:
            # Confidence first; on ties, the shorter faithful substring best
            # matches the containment-based Phrase F1 rule.
            result.append(max(choices, key=lambda item: (item[0], -item[1]))[2])
        else:
            result.append(phrase)
    return result


def _predict(frame, indices, grouped, risks, baselines, parameters):
    values = []
    for index in map(int, indices):
        if int(risks[index]) == config.RISK_LABELS["Indicator"]:
            values.append([]); continue
        text = str(frame.iloc[index].text)
        atomic = _atomic_candidates(
            text, grouped.get(index, []), parameters["token_threshold"],
            parameters["sentence_threshold"], parameters["max_tokens"],
        )
        values.append(_refine_one(text, baselines[index], atomic, parameters))
    return values


def _grid(frame, indices, grouped, risks, baselines):
    gold = [list(frame.iloc[int(index)].evidence) for index in indices]
    baseline_score = float(np.mean([
        _post_phrase_f1(baselines[int(index)], target)
        for index, target in zip(indices, gold)
    ]))
    rows = [{
        "mode": "baseline", "phrase_f1": baseline_score,
        "token_threshold": 1.0, "sentence_threshold": 1.0,
        "max_tokens": 4, "replacement_gate": 1.0,
        "minimum_length_ratio": 1.0, "minimum_baseline_tokens": 99,
        "alignment": "contained",
    }]
    atomic_cache = {}
    for token_threshold in (0.35, 0.45, 0.55, 0.65, 0.75):
        for sentence_threshold in (0.40, 0.60, 0.75):
            for max_tokens in (2, 3, 4, 6):
                cache_key = (token_threshold, sentence_threshold, max_tokens)
                atomic_cache[cache_key] = {}
                for index in map(int, indices):
                    if int(risks[index]) == config.RISK_LABELS["Indicator"]:
                        atomic_cache[cache_key][index] = []
                    else:
                        atomic_cache[cache_key][index] = _atomic_candidates(
                            str(frame.iloc[index].text), grouped.get(index, []),
                            token_threshold, sentence_threshold, max_tokens,
                        )
                # Candidate scores already include sqrt(sentence probability),
                # so this gate expresses the joint confidence.
                for replacement_gate in (0.55, 0.65, 0.75, 0.85):
                    for minimum_length_ratio in (0.25, 0.50, 0.75):
                        for minimum_baseline_tokens in (3, 5, 7):
                            for alignment in ("contained", "overlap"):
                                parameters = {
                                    "mode": "boundary_refine",
                                    "token_threshold": token_threshold,
                                    "sentence_threshold": sentence_threshold,
                                    "max_tokens": max_tokens,
                                    "replacement_gate": replacement_gate,
                                    "minimum_length_ratio": minimum_length_ratio,
                                    "minimum_baseline_tokens": minimum_baseline_tokens,
                                    "alignment": alignment,
                                }
                                predictions = []
                                for index in map(int, indices):
                                    if int(risks[index]) == config.RISK_LABELS["Indicator"]:
                                        predictions.append([])
                                    else:
                                        predictions.append(_refine_one(
                                            str(frame.iloc[index].text), baselines[index],
                                            atomic_cache[cache_key][index], parameters,
                                        ))
                                score = float(np.mean([
                                    _post_phrase_f1(prediction, target)
                                    for prediction, target in zip(predictions, gold)
                                ]))
                                rows.append({**parameters, "phrase_f1": score})
    rows.sort(key=lambda row: row["phrase_f1"], reverse=True)
    return rows


def _load_model(checkpoint_path, device):
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = AtomicEvidenceModel().to(device)
    model.load_state_dict(saved["state_dict"])
    return model


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("V26 requires CUDA")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    frame = load_train_data().reset_index(drop=True)
    labels = frame.risk_label.to_numpy()
    groups = frame.anon_user_id.astype(str).to_numpy()
    outer_train, outer_valid = next(StratifiedGroupKFold(
        n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED
    ).split(np.zeros(len(frame)), labels, groups))
    inner_records, membership, outer_raw = _load_records()
    inner_cal = np.asarray([
        index for index in outer_train if membership[int(index)] == INNER_CALIBRATION_FOLD
    ], dtype=np.int64)
    inner_by_index = {int(row["global_index"]): row for row in inner_records}
    outer_by_index = {
        int(index): record for index, record in zip(outer_valid, outer_raw["records"])
    }
    evidence_calibration = load_evidence_calibration()
    inner_risks = {int(index): int(inner_by_index[int(index)]["risk"]) for index in inner_cal}
    outer_risks = {index: int(row["risk"]) for index, row in outer_by_index.items()}
    inner_baselines = {
        int(index): _baseline_evidence(inner_by_index[int(index)], evidence_calibration)
        for index in inner_cal
    }
    outer_baselines = {
        index: _baseline_evidence(row, evidence_calibration)
        for index, row in outer_by_index.items()
    }
    tokenizer = AutoTokenizer.from_pretrained(
        config.MODEL_NAME, use_fast=True, local_files_only=True
    )
    cal_examples = _build_examples(frame, inner_cal, tokenizer, training=False)
    outer_examples = _build_examples(frame, outer_valid, tokenizer, training=False)
    inner_model = _load_model(INNER_CHECKPOINT, device)
    inner_outputs = _infer(inner_model, cal_examples, device, "V26 inner boundary calibration")
    del inner_model; torch.cuda.empty_cache()
    grid = _grid(frame, inner_cal, inner_outputs, inner_risks, inner_baselines)
    selected = grid[0]
    print(f"V26 selected on inner users: {selected}", flush=True)
    outer_model = _load_model(OUTER_CHECKPOINT, device)
    outer_outputs = _infer(outer_model, outer_examples, device, "V26 untouched outer boundary test")
    torch.save({
        "training_version": TRAINING_VERSION,
        "inner_outputs": dict(inner_outputs), "outer_outputs": dict(outer_outputs),
    }, RAW)
    predictions = _predict(
        frame, outer_valid, outer_outputs, outer_risks, outer_baselines, selected
    )
    baseline_values = np.asarray([
        _post_phrase_f1(outer_baselines[int(index)], list(frame.iloc[int(index)].evidence))
        for index in outer_valid
    ], dtype=np.float32)
    candidate_values = np.asarray([
        _post_phrase_f1(prediction, list(frame.iloc[int(index)].evidence))
        for prediction, index in zip(predictions, outer_valid)
    ], dtype=np.float32)
    truth = labels[outer_valid]
    risk = np.asarray([outer_risks[int(index)] for index in outer_valid])
    risk_f1 = float(f1_score(truth, risk, average="weighted", zero_division=0))
    baseline_task = task1_score(risk_f1, float(baseline_values.mean()))
    candidate_task = task1_score(risk_f1, float(candidate_values.mean()))
    bootstrap = _bootstrap(groups[outer_valid], baseline_values, candidate_values)
    adopted = bool(
        candidate_task >= baseline_task + 0.005
        and bootstrap["positive_fraction"] >= 0.80
    )
    payload = {
        "training_version": TRAINING_VERSION,
        "evaluation_scope": "boundary policy selected on inner users; one untouched outer evaluation",
        "selected": selected,
        "inner_top20": grid[:20],
        "baseline": {
            "risk_f1": risk_f1, "phrase_f1": float(baseline_values.mean()),
            "task1": baseline_task,
        },
        "candidate": {
            "risk_f1": risk_f1, "phrase_f1": float(candidate_values.mean()),
            "task1": candidate_task,
            "improved_posts": int((candidate_values > baseline_values).sum()),
            "worsened_posts": int((candidate_values < baseline_values).sum()),
        },
        "user_cluster_bootstrap": bootstrap,
        "adopted": adopted,
    }
    RESULTS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    CALIBRATION.write_text(json.dumps({
        "training_version": TRAINING_VERSION, "adopted": adopted,
        "strict_task1": candidate_task, "selected": selected,
    }, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    return payload


if __name__ == "__main__":
    main()
