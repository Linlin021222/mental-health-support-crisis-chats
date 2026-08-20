"""Boundary-aware semantic prototype continuation on strict PFA fold 0."""
from __future__ import annotations

import json

import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from transformers import AutoTokenizer

from configs.config import config
from inference.factor_nli import _rank_decode
from preprocess.factor_semantic_bank_v15 import (
    FACTOR_CLASSIFIER_PROMPTS, FACTOR_SEMANTIC_BANK, FIELDS,
)
from preprocess.preprocess import load_train_data
from trainer.factor_llm_lexical_v6 import _current_v3_probability
import trainer.factor_cross_encoder_v2 as v2


OUTPUT = config.OUTPUT_DIR / "factor_semantic_contrast_v15"
RESULTS = OUTPUT / "fold0_results.json"
TRAINING_VERSION = "factor-boundary-semantic-bank-v15"
REPLACEMENT_WEIGHT = 0.20


def train_fold0():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "semantic_bank.json").write_text(json.dumps([
        {"label": label, **entry}
        for label, entry in zip(config.FACTOR_LABELS, FACTOR_SEMANTIC_BANK)
    ], indent=2), encoding="utf-8")
    frame = load_train_data().reset_index(drop=True)
    targets = np.vstack(frame.factor_vector.to_numpy()).astype(np.int8)
    counts = np.vstack(frame.factor_counts.to_numpy()).astype(np.float32)
    risk = frame.risk_label.to_numpy(dtype=np.int64)
    groups = frame.anon_user_id.astype(str).to_numpy()
    fit_idx, valid_idx = next(StratifiedGroupKFold(
        n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED,
    ).split(np.zeros(len(frame)), risk, groups))

    # Reuse the thoroughly tested V2 MIL/hard-negative implementation, but
    # redirect every global to this isolated experiment and richer prompt bank.
    v2.FACTOR_PROTOTYPES = FACTOR_CLASSIFIER_PROMPTS
    v2.OUTPUT_DIR = OUTPUT
    v2.TRAINING_VERSION = TRAINING_VERSION
    tokenizer = AutoTokenizer.from_pretrained(
        config.FACTOR_NLI_MODEL_NAME, use_fast=True, local_files_only=True,
    )
    # V15 has more hypotheses than V3. Two 384-token pairs fit safely in the
    # 8 GB GPU and preserve the same effective batch size (2 * accumulation 8)
    # while two overlapping windows still cover roughly 640 source tokens.
    original_runtime = {
        "batch": config.FACTOR_PROTOTYPE_TRAIN_BATCH_SIZE,
        "accumulation": config.FACTOR_PROTOTYPE_ACCUMULATION,
        "length": config.FACTOR_NLI_MAX_LENGTH,
        "train_chunks": config.FACTOR_PROTOTYPE_TRAIN_MAX_CHUNKS,
        "infer_chunks": config.FACTOR_NLI_MAX_CHUNKS,
    }
    config.FACTOR_PROTOTYPE_TRAIN_BATCH_SIZE = 2
    config.FACTOR_PROTOTYPE_ACCUMULATION = 8
    config.FACTOR_NLI_MAX_LENGTH = 384
    config.FACTOR_PROTOTYPE_TRAIN_MAX_CHUNKS = 2
    config.FACTOR_NLI_MAX_CHUNKS = 2
    try:
        probability, training_summary = v2._train_fold(
            0, fit_idx, valid_idx, frame, targets, counts, tokenizer,
            torch.device(config.DEVICE),
        )
    finally:
        config.FACTOR_PROTOTYPE_TRAIN_BATCH_SIZE = original_runtime["batch"]
        config.FACTOR_PROTOTYPE_ACCUMULATION = original_runtime["accumulation"]
        config.FACTOR_NLI_MAX_LENGTH = original_runtime["length"]
        config.FACTOR_PROTOTYPE_TRAIN_MAX_CHUNKS = original_runtime["train_chunks"]
        config.FACTOR_NLI_MAX_CHUNKS = original_runtime["infer_chunks"]

    current, calibration = _current_v3_probability()
    v3_saved = np.load(config.OUTPUT_DIR / "factor_cross_encoder_v2" / "oof_predictions.npz")
    accepted_prototype = v3_saved["probabilities"][valid_idx]
    # Replace only 20% of the existing prototype component. Its production
    # component weight is ~0.476, so the new bank contributes ~9.5% overall.
    old_component_weight = float(calibration["new_cross_weight"])
    candidate = (current[valid_idx]
                 + old_component_weight * REPLACEMENT_WEIGHT
                 * (probability - accepted_prototype))
    prevalence = targets[fit_idx].mean(0)
    baseline_prediction = _rank_decode(current[valid_idx], prevalence, 1.10)
    candidate_prediction = _rank_decode(candidate, prevalence, 1.10)
    standalone_prediction = _rank_decode(probability, prevalence, 1.10)
    baseline = float(f1_score(targets[valid_idx], baseline_prediction,
                              average="macro", zero_division=0))
    candidate_score = float(f1_score(targets[valid_idx], candidate_prediction,
                                     average="macro", zero_division=0))
    standalone = float(f1_score(targets[valid_idx], standalone_prediction,
                                average="macro", zero_division=0))
    per_label = []
    for label, name in enumerate(config.FACTOR_LABELS):
        per_label.append({
            "label": name, "support": int(targets[valid_idx, label].sum()),
            "baseline_f1": float(f1_score(
                targets[valid_idx, label], baseline_prediction[:, label], zero_division=0)),
            "candidate_f1": float(f1_score(
                targets[valid_idx, label], candidate_prediction[:, label], zero_division=0)),
        })
    payload = {
        "training_version": TRAINING_VERSION,
        "semantic_bank": {"labels": len(FACTOR_SEMANTIC_BANK),
                          "fields_per_label": list(FIELDS),
                          "classifier_prompts_per_label": 5,
                          "review_basis": "published taxonomy plus train.xlsx chi-square phrases and confusion audit"},
        "training": training_summary,
        "fixed_policy": {"replace_fraction_of_v3_prototype_component": REPLACEMENT_WEIGHT,
                         "effective_total_weight": old_component_weight * REPLACEMENT_WEIGHT},
        "baseline_macro_f1": baseline, "semantic_cross_standalone_macro_f1": standalone,
        "candidate_macro_f1": candidate_score, "delta": candidate_score - baseline,
        "per_label": per_label,
        "promising_for_full_oof": bool(candidate_score >= baseline + .005),
        "adopted": False,
    }
    RESULTS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    return payload


if __name__ == "__main__":
    train_fold0()
