"""Frozen IMHI expert with long-text and sentence-prototype processing (V14).

The external encoder is never updated on PFA.  PFA classifiers are lightweight
label-specific logistic heads trained only on outer-training users.  Model-view
selection is performed with inner group OOF predictions; fusion weight and gate
are fixed before the untouched outer fold is scored.
"""
from __future__ import annotations

import json
import math
import re

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import normalize
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from configs.config import config
from inference.factor_nli import _rank_decode
from preprocess.factor_prototypes import FACTOR_PROTOTYPES
from preprocess.preprocess import load_train_data
from trainer.factor_external_transfer_v13 import EXTERNAL_ENCODER
from trainer.factor_llm_lexical_v6 import _current_v3_probability
from utils.seed import seed_everything


OUTPUT = config.OUTPUT_DIR / "factor_frozen_expert_v14"
FEATURES = OUTPUT / "features.npz"
RESULTS = OUTPUT / "fold0_results.json"
TRAINING_VERSION = "factor-frozen-imhi-sentence-expert-v14"
MAX_CHUNKS = 4
MAX_SENTENCES = 12
EXPERT_WEIGHT = 0.10
EXPERT_GATE_F1 = 0.35


def _sentences(text):
    text = str(text).strip()
    pieces = [x.strip() for x in re.split(r"(?<=[.!?])\s+|\n+", text) if x.strip()]
    if not pieces:
        pieces = [text]
    expanded = []
    for piece in pieces:
        words = piece.split()
        if len(words) <= 100:
            expanded.append(piece)
        else:
            expanded.extend(" ".join(words[i:i + 100]) for i in range(0, len(words), 100))
    if len(expanded) > MAX_SENTENCES:
        selected = np.linspace(0, len(expanded) - 1, MAX_SENTENCES).round().astype(int)
        expanded = [expanded[int(i)] for i in np.unique(selected)]
    return expanded


def _structural(text):
    text = str(text); words = text.split(); lowered = text.casefold()
    return np.asarray([
        math.log1p(len(words)), math.log1p(len(text)), math.log1p(text.count("\n")),
        math.log1p(text.count("!")), math.log1p(text.count("?")),
        sum(lowered.count(x) for x in (" i ", " i'm ", " me ", " my ")) / max(1, len(words)),
        sum(lowered.count(x) for x in (" no ", " not ", " never ", " nobody ", "nothing")) / max(1, len(words)),
        sum(lowered.count(x) for x in ("friend", "family", "partner", "therap", "doctor")) / max(1, len(words)),
        sum(lowered.count(x) for x in ("hope", "future", "goal", "purpose", "meaning")) / max(1, len(words)),
        sum(lowered.count(x) for x in ("alone", "lonely", "isolat", "abandon", "reject")) / max(1, len(words)),
    ], dtype=np.float32)


@torch.no_grad()
def _encode(model, tokenizer, texts, max_length, batch_size, device, desc):
    result = []
    for start in tqdm(range(0, len(texts), batch_size), desc=desc):
        batch = texts[start:start + batch_size]
        encoded = tokenizer(batch, padding=True, truncation=True, max_length=max_length,
                            return_tensors="pt")
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            hidden = model(**encoded).last_hidden_state.float()
        weight = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * weight).sum(1) / weight.sum(1).clamp_min(1.0)
        result.append(torch.nn.functional.normalize(pooled, dim=-1).cpu().numpy())
    return np.vstack(result).astype(np.float32)


def _chunk_texts(tokenizer, text):
    encoded = tokenizer(
        str(text), truncation=True, max_length=256, stride=64,
        return_overflowing_tokens=True, add_special_tokens=True,
    )["input_ids"]
    if len(encoded) > MAX_CHUNKS:
        selected = np.linspace(0, len(encoded) - 1, MAX_CHUNKS).round().astype(int)
        encoded = [encoded[int(i)] for i in np.unique(selected)]
    return [tokenizer.decode(ids, skip_special_tokens=True) for ids in encoded]


def extract_features(frame):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    row_ids = frame.row_id.astype(str).to_numpy()
    if FEATURES.exists():
        saved = np.load(FEATURES)
        if np.array_equal(saved["row_ids"].astype(str), row_ids):
            print(f"V14 loaded cached features: {FEATURES}", flush=True)
            return saved["first"], saved["processed"], json.loads(str(saved["audit"]))
    if not EXTERNAL_ENCODER.exists():
        raise FileNotFoundError(f"Run --mode factor-external-transfer-v13-fold0 first: {EXTERNAL_ENCODER}")
    device = torch.device(config.DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(config.FACTOR_MODEL_NAME, use_fast=True)
    model = AutoModel.from_pretrained(config.FACTOR_MODEL_NAME, dtype=torch.float32)
    model.load_state_dict(torch.load(EXTERNAL_ENCODER, map_location="cpu", weights_only=True), strict=False)
    model = model.to(device).eval()

    texts = frame.text.astype(str).tolist()
    chunk_lists = [_chunk_texts(tokenizer, text) for text in tqdm(texts, desc="V14 chunking")]
    chunk_text, chunk_owner = [], []
    for owner, chunks in enumerate(chunk_lists):
        chunk_text.extend(chunks); chunk_owner.extend([owner] * len(chunks))
    chunk_embedding = _encode(model, tokenizer, chunk_text, 256, 16, device, "V14 chunks")
    chunk_owner = np.asarray(chunk_owner)
    first = np.zeros((len(texts), chunk_embedding.shape[1]), dtype=np.float32)
    mean = np.zeros_like(first); maximum = np.zeros_like(first)
    chunk_counts = []
    for owner in range(len(texts)):
        values = chunk_embedding[chunk_owner == owner]
        first[owner] = values[0]; mean[owner] = values.mean(0); maximum[owner] = values.max(0)
        chunk_counts.append(len(values))

    sentence_lists = [_sentences(text) for text in texts]
    sentence_text, sentence_owner = [], []
    for owner, sentences in enumerate(sentence_lists):
        sentence_text.extend(sentences); sentence_owner.extend([owner] * len(sentences))
    sentence_embedding = _encode(model, tokenizer, sentence_text, 128, 24, device, "V14 sentences")
    sentence_owner = np.asarray(sentence_owner)

    flat_prototypes = [text for descriptions in FACTOR_PROTOTYPES for text in descriptions]
    prototype_embedding = _encode(model, tokenizer, flat_prototypes, 96, 24, device, "V14 prototypes")
    prototype_embedding = prototype_embedding.reshape(config.NUM_FACTORS, 3, -1)
    similarity = np.zeros((len(texts), config.NUM_FACTORS), dtype=np.float32)
    for owner in range(len(texts)):
        values = sentence_embedding[sentence_owner == owner]
        # max over sentences and over the three positive descriptions
        score = np.einsum("sh,kph->skp", values, prototype_embedding)
        similarity[owner] = score.max(axis=(0, 2))

    structure = np.vstack([_structural(text) for text in texts])
    structure = (structure - structure.mean(0)) / (structure.std(0) + 1e-6)
    processed = np.concatenate([normalize(mean), normalize(maximum), similarity, structure], axis=1)
    processed = normalize(processed).astype(np.float32)
    first = normalize(first).astype(np.float32)
    audit = {
        "posts": len(texts), "total_selected_chunks": len(chunk_text),
        "posts_with_multiple_chunks": int(np.sum(np.asarray(chunk_counts) > 1)),
        "posts_requiring_four_chunks": int(np.sum(np.asarray(chunk_counts) == MAX_CHUNKS)),
        "total_selected_sentences": len(sentence_text), "max_sentences": MAX_SENTENCES,
        "processed_feature_dimensions": int(processed.shape[1]),
    }
    np.savez_compressed(FEATURES, row_ids=row_ids, first=first, processed=processed,
                        audit=json.dumps(audit))
    del model
    if device.type == "cuda": torch.cuda.empty_cache()
    return first, processed, audit


def _models_predict(features, targets, train_idx, valid_idx):
    probability = np.zeros((len(valid_idx), config.NUM_FACTORS), dtype=np.float32)
    for label in range(config.NUM_FACTORS):
        y = targets[train_idx, label]
        if y.min() == y.max():
            probability[:, label] = float(y[0])
            continue
        model = LogisticRegression(
            C=.05, solver="liblinear", class_weight="balanced", max_iter=1500,
            random_state=config.SEED + label,
        )
        model.fit(features[train_idx], y)
        probability[:, label] = model.predict_proba(features[valid_idx])[:, 1]
    return probability


def _rank_columns(values):
    values = np.asarray(values)
    result = np.empty_like(values, dtype=np.float32)
    for label in range(values.shape[1]):
        order = np.argsort(np.argsort(values[:, label], kind="mergesort"), kind="mergesort")
        result[:, label] = (order + .5) / len(values)
    return result


def train_fold0():
    seed_everything(config.SEED + 1414); OUTPUT.mkdir(parents=True, exist_ok=True)
    frame = load_train_data().reset_index(drop=True)
    targets = np.vstack(frame.factor_vector.to_numpy()).astype(np.int8)
    risk = frame.risk_label.to_numpy(dtype=np.int64)
    groups = frame.anon_user_id.astype(str).to_numpy()
    first, processed, audit = extract_features(frame)
    fit_idx, valid_idx = next(StratifiedGroupKFold(
        n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED,
    ).split(np.zeros(len(frame)), risk, groups))
    prevalence = targets[fit_idx].mean(0)

    inner = list(StratifiedGroupKFold(
        n_splits=4, shuffle=True, random_state=config.SEED + 1414,
    ).split(np.zeros(len(fit_idx)), risk[fit_idx], groups[fit_idx]))
    first_oof = np.zeros((len(fit_idx), config.NUM_FACTORS), dtype=np.float32)
    processed_oof = np.zeros_like(first_oof)
    for fold, (inner_train, inner_valid) in enumerate(inner):
        train_global, valid_global = fit_idx[inner_train], fit_idx[inner_valid]
        first_oof[inner_valid] = _models_predict(first, targets, train_global, valid_global)
        processed_oof[inner_valid] = _models_predict(processed, targets, train_global, valid_global)
        print(f"V14 inner fold {fold + 1}/4", flush=True)

    selected_processed = np.zeros(config.NUM_FACTORS, dtype=bool)
    selected_score = np.zeros(config.NUM_FACTORS, dtype=np.float32)
    for label in range(config.NUM_FACTORS):
        truth = targets[fit_idx, label]
        first_prediction = _rank_decode(
            first_oof[:, [label]], prevalence[[label]], 1.10)[:, 0]
        processed_prediction = _rank_decode(
            processed_oof[:, [label]], prevalence[[label]], 1.10)[:, 0]
        first_score = f1_score(truth, first_prediction, zero_division=0)
        processed_score = f1_score(truth, processed_prediction, zero_division=0)
        selected_processed[label] = processed_score > first_score
        selected_score[label] = max(first_score, processed_score)

    first_valid = _models_predict(first, targets, fit_idx, valid_idx)
    processed_valid = _models_predict(processed, targets, fit_idx, valid_idx)
    expert_valid = np.where(selected_processed[None, :], processed_valid, first_valid)
    gate = selected_score >= EXPERT_GATE_F1

    base, _ = _current_v3_probability(); base_valid = base[valid_idx]
    baseline_prediction = _rank_decode(base_valid, prevalence, 1.10)
    base_rank = _rank_columns(base_valid); expert_rank = _rank_columns(expert_valid)
    candidate_rank = base_rank.copy()
    candidate_rank[:, gate] = ((1.0 - EXPERT_WEIGHT) * base_rank[:, gate]
                               + EXPERT_WEIGHT * expert_rank[:, gate])
    candidate_prediction = _rank_decode(candidate_rank, prevalence, 1.10)

    # Predeclared ablations quantify the value of data processing. They are
    # reported but cannot change the fixed candidate after seeing outer labels.
    first_rank = _rank_columns(first_valid)
    first_mix = (1.0 - EXPERT_WEIGHT) * base_rank + EXPERT_WEIGHT * first_rank
    first_prediction = _rank_decode(first_mix, prevalence, 1.10)
    baseline = float(f1_score(targets[valid_idx], baseline_prediction,
                              average="macro", zero_division=0))
    candidate = float(f1_score(targets[valid_idx], candidate_prediction,
                               average="macro", zero_division=0))
    first_score = float(f1_score(targets[valid_idx], first_prediction,
                                 average="macro", zero_division=0))
    expert_standalone = float(f1_score(
        targets[valid_idx], _rank_decode(expert_valid, prevalence, 1.10),
        average="macro", zero_division=0))
    per_label=[]
    for label,name in enumerate(config.FACTOR_LABELS):
        per_label.append({
            "label":name,"support":int(targets[valid_idx,label].sum()),
            "selected_view":"processed" if selected_processed[label] else "first256",
            "inner_expert_f1":float(selected_score[label]),"gate":bool(gate[label]),
            "baseline_f1":float(f1_score(targets[valid_idx,label],baseline_prediction[:,label],zero_division=0)),
            "candidate_f1":float(f1_score(targets[valid_idx,label],candidate_prediction[:,label],zero_division=0)),
        })
    payload={
        "training_version":TRAINING_VERSION,
        "evaluation":"untouched outer user fold; view selected on inner group OOF",
        "data_processing_audit":audit,
        "fixed_policy":{"expert_weight":EXPERT_WEIGHT,"inner_f1_gate":EXPERT_GATE_F1},
        "baseline_macro_f1":baseline,"first256_fixed_blend_macro_f1":first_score,
        "processed_gated_macro_f1":candidate,"expert_standalone_macro_f1":expert_standalone,
        "delta":candidate-baseline,"gated_labels":int(gate.sum()),"per_label":per_label,
        "promising_for_full_oof":bool(candidate>=baseline+.005),"adopted":False,
    }
    RESULTS.write_text(json.dumps(payload,indent=2),encoding="utf-8")
    print(json.dumps(payload,indent=2),flush=True); return payload


if __name__ == "__main__":
    train_fold0()
