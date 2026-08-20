"""Definition-grounded residual ranker for Task 2 (strict fold 0).

V23 tests a deliberately small, interpretable change to the accepted
MentalRoBERTa factor encoder.  Each post token is compared with four positive
descriptions and two boundary descriptions for every factor.  The resulting
multi-scale relevance statistics are learned together with the transferred
factor logit by a strongly regularised per-label ranker.

The outer user-disjoint fold is used exactly once for reporting.  The decoder
and the contribution of the new ranker are fixed in advance, so a change in
prevalence or threshold cannot be mistaken for a better representation.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from transformers import AutoTokenizer

from configs.config import config
from datasets.factor_cache_builder import build_factor_cache
from inference.factor_nli import _rank_decode
from models.factor_model import MentalRobertaFactorModel
from preprocess.factor_semantic_bank_v15 import FACTOR_SEMANTIC_BANK
from preprocess.preprocess import load_train_data
from trainer.factor_llm_lexical_v6 import _current_v3_probability
from trainer.factor_train import FactorDataset, _loader
from utils.seed import seed_everything


OUTPUT = config.OUTPUT_DIR / "factor_definition_ranker_v23"
FEATURES = OUTPUT / "fold0_features.npz"
RESULTS = OUTPUT / "fold0_results.json"
TRAINING_VERSION = "definition-multiscale-residual-ranker-v23b"

# Fixed before looking at the outer holdout.  The accepted semantic component
# is changed by only 25%; because that component has a 0.7 weight inside the
# base and the base has a ~0.38 weight in V3, total contribution is ~6.7%.
SEMANTIC_REPLACEMENT = 0.25
LOGISTIC_C = 0.10
TOP_K = (1, 3, 8)


def _mean_pool(hidden, mask):
    weight = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * weight).sum(1) / weight.sum(1).clamp_min(1.0)


@torch.no_grad()
def _encode_definition_bank(model, tokenizer, device):
    positive, boundary = [], []
    for label, entry in zip(config.FACTOR_LABELS, FACTOR_SEMANTIC_BANK):
        positive.append([
            f"Factor {label}. {entry[field]}"
            for field in ("formal", "direct", "implicit", "train_summary")
        ])
        boundary.append([
            f"Boundary for {label}. {entry['distinction']}",
            f"Not sufficient evidence for {label}. {entry['negative']}",
        ])

    flat = [text for rows in positive + boundary for text in rows]
    # The two banks have different prototype counts, so encode them separately
    # after a shared transformer pass over the flattened definition strings.
    encoded_rows = []
    for start in range(0, len(flat), 24):
        tokenized = tokenizer(
            flat[start:start + 24], padding=True, truncation=True,
            max_length=112, return_tensors="pt",
        )
        tokenized = {key: value.to(device) for key, value in tokenized.items()}
        hidden = model.encoder(**tokenized).last_hidden_state.float()
        encoded_rows.append(_mean_pool(hidden, tokenized["attention_mask"]))
    values = F.normalize(torch.cat(encoded_rows), dim=-1)
    positive_count = config.NUM_FACTORS * 4
    positive_values = values[:positive_count].reshape(config.NUM_FACTORS, 4, -1)
    boundary_values = values[positive_count:].reshape(config.NUM_FACTORS, 2, -1)
    return positive_values, boundary_values


def _top_mean(similarity, valid_mask, k):
    """Top-k token relevance for [batch, token, label] similarities."""
    masked = similarity.masked_fill(~valid_mask.unsqueeze(-1), -1e4)
    usable = min(k, masked.size(1))
    values = torch.topk(masked, k=usable, dim=1).values
    # Every real post has at least one non-special token. Padding chunks are
    # masked and therefore cannot enter the top-k unless a malformed row is
    # completely empty.
    return values.mean(1)


@torch.no_grad()
def _batch_features(model, ids, mask, positive, boundary, special_ids):
    batch, chunks, length = ids.shape
    flat_ids = ids.reshape(batch * chunks, length)
    flat_mask = mask.reshape(batch * chunks, length)
    hidden = model.encoder(input_ids=flat_ids, attention_mask=flat_mask).last_hidden_state
    tokens = model.norm(hidden.float().reshape(batch, chunks * length, -1))
    base_mask = mask.reshape(batch, chunks * length).bool()
    valid = base_mask.clone()
    for token_id in special_ids:
        valid &= ids.reshape(batch, chunks * length).ne(int(token_id))

    # Reproduce the transferred one-hop MentalRoBERTa logits exactly while the
    # same encoder output is already in memory.
    scale = math.sqrt(tokens.size(-1))
    old_scores = torch.einsum("bth,kh->btk", tokens, model.label_queries) / scale
    old_scores = old_scores.masked_fill(~base_mask.unsqueeze(-1), -1e4)
    old_attention = torch.softmax(old_scores, dim=1)
    old_repr = torch.einsum("btk,bth->bkh", old_attention, tokens)
    local = (old_repr * model.label_weights.unsqueeze(0)).sum(-1) + model.label_bias
    base_float = base_mask.unsqueeze(-1).to(tokens.dtype)
    global_repr = (tokens * base_float).sum(1) / base_float.sum(1).clamp_min(1.0)
    global_logits = torch.cat(
        [model.global_risk(global_repr), model.global_protective(global_repr)], dim=-1
    )
    old_logits = local + global_logits

    normal_tokens = F.normalize(tokens, dim=-1)
    positive_similarity = torch.einsum("bth,kph->btkp", normal_tokens, positive)
    boundary_similarity = torch.einsum("bth,kph->btkp", normal_tokens, boundary)
    positive_token = positive_similarity.max(-1).values
    boundary_token = boundary_similarity.max(-1).values

    features = []
    for k in TOP_K:
        features.append(_top_mean(positive_token, valid, k))
    # Coverage asks whether each distinct positive description can find a
    # supporting token, instead of letting one keyword satisfy every prompt.
    positive_masked = positive_similarity.masked_fill(
        ~valid.unsqueeze(-1).unsqueeze(-1), -1e4
    )
    boundary_masked = boundary_similarity.masked_fill(
        ~valid.unsqueeze(-1).unsqueeze(-1), -1e4
    )
    positive_coverage = positive_masked.max(1).values.mean(-1)
    boundary_coverage = boundary_masked.max(1).values.mean(-1)
    features.append(positive_coverage)
    for k in TOP_K:
        features.append(_top_mean(boundary_token, valid, k))
    features.append(boundary_coverage)

    # A chunk-level view is less sensitive to one isolated keyword and helps
    # long Reddit posts where the supporting context spans several tokens.
    chunk_tokens = tokens.reshape(batch, chunks, length, -1)
    chunk_mask = valid.reshape(batch, chunks, length)
    chunk_weight = chunk_mask.unsqueeze(-1).to(tokens.dtype)
    chunk_repr = (chunk_tokens * chunk_weight).sum(2) / chunk_weight.sum(2).clamp_min(1.0)
    chunk_repr = F.normalize(chunk_repr, dim=-1)
    chunk_positive = torch.einsum("bch,kph->bckp", chunk_repr, positive).amax((1, 3))
    chunk_boundary = torch.einsum("bch,kph->bckp", chunk_repr, boundary).amax((1, 3))
    features.extend([chunk_positive, chunk_boundary])

    # Explicit margins make the exclusion rules available to a linear ranker.
    features.extend([
        features[0] - features[4],       # strongest positive vs boundary token
        positive_coverage - boundary_coverage,
        chunk_positive - chunk_boundary,
    ])
    # Transferred task logit is the anchor; the other 13 dimensions explain
    # why that score should move up or down under the full factor definition.
    stacked = torch.stack([old_logits, *features], dim=-1)
    return old_logits, stacked


@torch.no_grad()
def _extract_features(dataset, model, tokenizer, device, indices=None):
    if indices is None:
        indices = np.arange(len(dataset))
    loader = _loader(dataset, np.asarray(indices, dtype=np.int64), False)
    positive, boundary = _encode_definition_bank(model, tokenizer, device)
    special = set(tokenizer.all_special_ids)
    all_logits, all_features = [], []
    model.eval()
    for batch in tqdm(loader, desc="V23 definition-token relevance"):
        logits, features = _batch_features(
            model,
            batch["input_ids"].to(device, non_blocking=True),
            batch["attention_mask"].to(device, non_blocking=True),
            positive, boundary, special,
        )
        all_logits.append(logits.cpu().numpy())
        all_features.append(features.cpu().numpy())
    return np.vstack(all_logits).astype(np.float32), np.vstack(all_features).astype(np.float32)


def _fit_rankers(features, targets, train_idx, valid_idx):
    probabilities = np.zeros((len(valid_idx), config.NUM_FACTORS), dtype=np.float32)
    coefficients = []
    for label in range(config.NUM_FACTORS):
        x_train = features[train_idx, label]
        x_valid = features[valid_idx, label]
        y_train = targets[train_idx, label]
        scaler = StandardScaler().fit(x_train)
        if np.unique(y_train).size < 2:
            probabilities[:, label] = float(y_train.mean())
            coefficients.append({"constant": float(y_train.mean())})
            continue
        ranker = LogisticRegression(
            C=LOGISTIC_C, class_weight="balanced", solver="liblinear",
            max_iter=2000, random_state=config.SEED + label,
        )
        ranker.fit(scaler.transform(x_train), y_train)
        probabilities[:, label] = ranker.predict_proba(scaler.transform(x_valid))[:, 1]
        coefficients.append({
            "intercept": float(ranker.intercept_[0]),
            "weights": ranker.coef_[0].astype(float).tolist(),
        })
    return probabilities, coefficients


def _safe_macro_auc(targets, probability):
    values = []
    for label in range(targets.shape[1]):
        if np.unique(targets[:, label]).size == 2:
            values.append(roc_auc_score(targets[:, label], probability[:, label]))
    return float(np.mean(values))


def train_fold0(force_features=False):
    if not torch.cuda.is_available():
        raise RuntimeError("Factor definition ranker V23 requires CUDA")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    seed_everything(config.SEED + 2300)
    frame = load_train_data().reset_index(drop=True)
    targets = np.vstack(frame.factor_vector.to_numpy()).astype(np.int8)
    risk = frame.risk_label.to_numpy(dtype=np.int64)
    groups = frame.anon_user_id.astype(str).to_numpy()
    train_idx, valid_idx = next(StratifiedGroupKFold(
        n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED,
    ).split(np.zeros(len(frame)), risk, groups))

    source = config.OUTPUT_DIR / "factor_cv" / "fold0_model.pt"
    if not source.exists():
        raise FileNotFoundError(f"Missing accepted Task 2 checkpoint: {source}")
    if force_features or not FEATURES.exists():
        cache = config.CACHE_DIR / "factor_train_cache.pt"
        if not cache.exists():
            cache = build_factor_cache(train=True)
        dataset = FactorDataset(cache)
        device = torch.device("cuda")
        model = MentalRobertaFactorModel(initialise_labels=False).to(device)
        model.load_state_dict(torch.load(source, map_location="cpu"))
        tokenizer = AutoTokenizer.from_pretrained(
            config.FACTOR_MODEL_NAME, use_fast=True, local_files_only=True,
        )
        logits, features = _extract_features(dataset, model, tokenizer, device)
        np.savez_compressed(
            FEATURES, logits=logits, features=features,
            training_version=TRAINING_VERSION,
        )
        del model, dataset
        torch.cuda.empty_cache()
    else:
        saved = np.load(FEATURES)
        if str(saved["training_version"]) != TRAINING_VERSION:
            raise RuntimeError("Stale V23 feature cache; rerun with force_features=True")
        logits = saved["logits"].astype(np.float32)
        features = saved["features"].astype(np.float32)
        print(f"Loaded cached V23 features: {FEATURES}", flush=True)

    ranker_probability, coefficients = _fit_rankers(
        features, targets, train_idx, valid_idx
    )
    old_semantic = np.load(
        config.OUTPUT_DIR / "factor_cv" / "factor_oof_predictions.npz"
    )["semantic"].astype(np.float32)
    transferred_probability = torch.sigmoid(torch.from_numpy(logits)).numpy()
    # Guard against an accidental mismatch in token masking or checkpoint.
    reproduction_mae = float(np.abs(
        transferred_probability[valid_idx] - old_semantic[valid_idx]
    ).mean())

    upgraded_semantic = (
        (1.0 - SEMANTIC_REPLACEMENT) * old_semantic[valid_idx]
        + SEMANTIC_REPLACEMENT * ranker_probability
    )
    current, calibration = _current_v3_probability()
    base_weight = float(calibration["base_weight"])
    semantic_weight = float(config.FACTOR_SEMANTIC_MODEL_WEIGHT)
    candidate_probability = (
        current[valid_idx]
        + base_weight * semantic_weight
        * (upgraded_semantic - old_semantic[valid_idx])
    )
    prevalence = targets[train_idx].mean(0)
    baseline_prediction = _rank_decode(current[valid_idx], prevalence, 1.10)
    candidate_prediction = _rank_decode(candidate_probability, prevalence, 1.10)
    ranker_prediction = _rank_decode(ranker_probability, prevalence, 1.10)
    baseline = float(f1_score(
        targets[valid_idx], baseline_prediction, average="macro", zero_division=0
    ))
    candidate = float(f1_score(
        targets[valid_idx], candidate_prediction, average="macro", zero_division=0
    ))
    standalone = float(f1_score(
        targets[valid_idx], ranker_prediction, average="macro", zero_division=0
    ))
    per_label = []
    for label, name in enumerate(config.FACTOR_LABELS):
        per_label.append({
            "label": name,
            "support": int(targets[valid_idx, label].sum()),
            "baseline_f1": float(f1_score(
                targets[valid_idx, label], baseline_prediction[:, label], zero_division=0
            )),
            "candidate_f1": float(f1_score(
                targets[valid_idx, label], candidate_prediction[:, label], zero_division=0
            )),
            "old_semantic_auc": (
                float(roc_auc_score(targets[valid_idx, label], old_semantic[valid_idx, label]))
                if np.unique(targets[valid_idx, label]).size == 2 else None
            ),
            "definition_ranker_auc": (
                float(roc_auc_score(targets[valid_idx, label], ranker_probability[:, label]))
                if np.unique(targets[valid_idx, label]).size == 2 else None
            ),
        })
    payload = {
        "training_version": TRAINING_VERSION,
        "evaluation": "untouched outer StratifiedGroupKFold fold 0; fixed ratio=1.10",
        "architecture": {
            "positive_prototypes_per_label": 4,
            "boundary_prototypes_per_label": 2,
            "feature_views": ["token_top1", "token_top3", "token_top8",
                              "prototype_coverage", "chunk_context", "boundary_margins"],
            "feature_dimensions_per_label": int(features.shape[-1]),
            "logistic_c": LOGISTIC_C,
            "semantic_replacement": SEMANTIC_REPLACEMENT,
            "effective_total_weight": base_weight * semantic_weight * SEMANTIC_REPLACEMENT,
        },
        "checkpoint_reproduction_mae": reproduction_mae,
        "baseline_macro_f1": baseline,
        "definition_ranker_standalone_macro_f1": standalone,
        "candidate_macro_f1": candidate,
        "delta": candidate - baseline,
        "old_semantic_macro_auc": _safe_macro_auc(
            targets[valid_idx], old_semantic[valid_idx]
        ),
        "definition_ranker_macro_auc": _safe_macro_auc(
            targets[valid_idx], ranker_probability
        ),
        "promising_for_full_oof": bool(candidate >= baseline + .005),
        "adopted": False,
        "per_label": per_label,
        "ranker_coefficients": coefficients,
    }
    RESULTS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    return payload


if __name__ == "__main__":
    train_fold0()
