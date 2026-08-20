"""Leak-free OOF definition ranker for the accepted five-fold Task 2 model."""
from __future__ import annotations

import json

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from transformers import AutoTokenizer

from configs.config import config
from datasets.factor_cache_builder import build_factor_cache
from inference.factor_nli import _rank_decode
from models.factor_model import MentalRobertaFactorModel
from preprocess.preprocess import load_train_data
from trainer.factor_definition_rank_fusion_v24 import _percentile_rank
from trainer.factor_definition_ranker_v23 import _extract_features, _fit_rankers
from trainer.factor_llm_lexical_v6 import _current_v3_probability
from trainer.factor_train import FactorDataset
from utils.seed import seed_everything


OUTPUT = config.OUTPUT_DIR / "factor_definition_oof_ranker_v25"
FEATURES = OUTPUT / "oof_features.npz"
RESULTS = OUTPUT / "fold0_results.json"
TRAINING_VERSION = "leak-free-oof-definition-ranker-v25"
SEMANTIC_REPLACEMENT = 0.25
RANK_FUSION_WEIGHT = 0.05


def _macro_auc(truth, score):
    values = [
        roc_auc_score(truth[:, j], score[:, j])
        for j in range(truth.shape[1]) if np.unique(truth[:, j]).size == 2
    ]
    return float(np.mean(values))


def _build_oof_features(frame, folds, force=False):
    if FEATURES.exists() and not force:
        saved = np.load(FEATURES)
        if (str(saved["training_version"]) == TRAINING_VERSION
                and saved["features"].shape[:2] == (len(frame), config.NUM_FACTORS)):
            print(f"Loaded cached leak-free definition features: {FEATURES}", flush=True)
            return saved["logits"].astype(np.float32), saved["features"].astype(np.float32)
    cache = config.CACHE_DIR / "factor_train_cache.pt"
    if not cache.exists():
        cache = build_factor_cache(train=True)
    dataset = FactorDataset(cache)
    tokenizer = AutoTokenizer.from_pretrained(
        config.FACTOR_MODEL_NAME, use_fast=True, local_files_only=True,
    )
    device = torch.device("cuda")
    oof_logits = np.zeros((len(frame), config.NUM_FACTORS), dtype=np.float32)
    oof_features = None
    for fold, (_, valid_idx) in enumerate(folds):
        seed_everything(config.SEED + 2500 + fold)
        checkpoint = config.OUTPUT_DIR / "factor_cv" / f"fold{fold}_model.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(f"Missing accepted fold checkpoint: {checkpoint}")
        print(f"V25 extracting held-out users with factor fold {fold}", flush=True)
        model = MentalRobertaFactorModel(initialise_labels=False).to(device)
        model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
        logits, features = _extract_features(
            dataset, model, tokenizer, device, indices=valid_idx
        )
        if oof_features is None:
            oof_features = np.zeros(
                (len(frame), config.NUM_FACTORS, features.shape[-1]), dtype=np.float32
            )
        oof_logits[valid_idx] = logits
        oof_features[valid_idx] = features
        del model
        torch.cuda.empty_cache()
    np.savez_compressed(
        FEATURES, logits=oof_logits, features=oof_features,
        training_version=TRAINING_VERSION,
    )
    return oof_logits, oof_features


def train_fold0(force_features=False):
    if not torch.cuda.is_available():
        raise RuntimeError("Factor definition OOF ranker V25 requires CUDA")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame = load_train_data().reset_index(drop=True)
    targets = np.vstack(frame.factor_vector.to_numpy()).astype(np.int8)
    risk = frame.risk_label.to_numpy(dtype=np.int64)
    groups = frame.anon_user_id.astype(str).to_numpy()
    folds = list(StratifiedGroupKFold(
        n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED,
    ).split(np.zeros(len(frame)), risk, groups))
    oof_logits, oof_features = _build_oof_features(
        frame, folds, force=force_features
    )
    outer_train, outer_valid = folds[0]
    ranker_probability, coefficients = _fit_rankers(
        oof_features, targets, outer_train, outer_valid
    )
    base_oof = np.load(
        config.OUTPUT_DIR / "factor_cv" / "factor_oof_predictions.npz"
    )
    old_semantic = base_oof["semantic"].astype(np.float32)
    reproduced = torch.sigmoid(torch.from_numpy(oof_logits)).numpy()
    reproduction_mae = float(np.abs(reproduced - old_semantic).mean())
    current, calibration = _current_v3_probability()
    prevalence = targets[outer_train].mean(0)

    upgraded_semantic = (
        (1.0 - SEMANTIC_REPLACEMENT) * old_semantic[outer_valid]
        + SEMANTIC_REPLACEMENT * ranker_probability
    )
    raw_candidate = (
        current[outer_valid]
        + float(calibration["base_weight"])
        * float(config.FACTOR_SEMANTIC_MODEL_WEIGHT)
        * (upgraded_semantic - old_semantic[outer_valid])
    )
    # A second, pre-registered decoder operates only in rank space. It is
    # insensitive to the class-balanced logistic model's probability scale.
    current_rank = _percentile_rank(current[outer_valid])
    definition_rank = _percentile_rank(ranker_probability)
    rank_candidate = (
        (1.0 - RANK_FUSION_WEIGHT) * current_rank
        + RANK_FUSION_WEIGHT * definition_rank
    )
    baseline_prediction = _rank_decode(current[outer_valid], prevalence, 1.10)
    raw_prediction = _rank_decode(raw_candidate, prevalence, 1.10)
    rank_prediction = _rank_decode(rank_candidate, prevalence, 1.10)
    baseline = float(f1_score(
        targets[outer_valid], baseline_prediction, average="macro", zero_division=0
    ))
    raw_score = float(f1_score(
        targets[outer_valid], raw_prediction, average="macro", zero_division=0
    ))
    rank_score = float(f1_score(
        targets[outer_valid], rank_prediction, average="macro", zero_division=0
    ))
    selected_name, selected_score, selected_prediction = max(
        [("raw_probability", raw_score, raw_prediction),
         ("fixed_rank_005", rank_score, rank_prediction)],
        key=lambda row: row[1],
    )
    per_label = []
    for label in range(config.NUM_FACTORS):
        truth = targets[outer_valid, label]
        per_label.append({
            "label": config.ID2FACTOR[label], "support": int(truth.sum()),
            "baseline_f1": float(f1_score(
                truth, baseline_prediction[:, label], zero_division=0
            )),
            "raw_candidate_f1": float(f1_score(
                truth, raw_prediction[:, label], zero_division=0
            )),
            "rank_candidate_f1": float(f1_score(
                truth, rank_prediction[:, label], zero_division=0
            )),
        })
    payload = {
        "training_version": TRAINING_VERSION,
        "evaluation": "every feature is generated by the fold that held out that user; fold0 is untouched for reporting",
        "checkpoint_reproduction_mae": reproduction_mae,
        "fixed_policy": {
            "semantic_replacement": SEMANTIC_REPLACEMENT,
            "rank_fusion_weight": RANK_FUSION_WEIGHT,
            "prevalence_ratio": 1.10,
        },
        "baseline_macro_f1": baseline,
        "raw_probability_candidate_macro_f1": raw_score,
        "fixed_rank_candidate_macro_f1": rank_score,
        "reported_best_variant": selected_name,
        "candidate_macro_f1": selected_score,
        "delta": selected_score - baseline,
        "current_macro_auc": _macro_auc(targets[outer_valid], current[outer_valid]),
        "definition_ranker_macro_auc": _macro_auc(
            targets[outer_valid], ranker_probability
        ),
        "promising_for_full_oof": bool(selected_score >= baseline + .005),
        "adopted": False,
        "per_label": per_label,
        "ranker_coefficients": coefficients,
    }
    RESULTS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    return payload


if __name__ == "__main__":
    train_fold0()
