"""Strict prototype/window-pooling ablation for the accepted Task 2 v3 folds.

No checkpoint is retrained.  The script exposes scores that prototype-MIL-v3
currently discards, then compares pooling rules on the identical user-held-out
fold.  The accepted mean-prototype/max-window rule is included as an exact
reproduction check.
"""
from __future__ import annotations

import json

import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from configs.config import config
from inference.factor_nli import _entailment_index, _rank_decode
from preprocess.factor_prototypes import FACTOR_PROTOTYPES
from preprocess.preprocess import load_train_data


OUTPUT_DIR = config.OUTPUT_DIR / "factor_pooling_v5"
V3_DIR = config.OUTPUT_DIR / "factor_cross_encoder_v2"


def _prototype_pool(values, mode):
    if mode == "mean":
        return values.mean(1)
    if mode == "top2":
        return np.sort(values, axis=1)[:, -min(2, values.shape[1]):].mean(1)
    if mode == "max":
        return values.max(1)
    if mode == "logit_mean":
        clipped = np.clip(values, 1e-5, 1.0 - 1e-5)
        logits = np.log(clipped / (1.0 - clipped)).mean(1)
        return 1.0 / (1.0 + np.exp(-logits))
    raise ValueError(mode)


@torch.no_grad()
def _predict_variants(model, tokenizer, texts, device):
    model.eval(); entailment = _entailment_index(model)
    names = [
        "window_max__prototype_mean", "window_max__prototype_top2",
        "window_max__prototype_max", "window_max__prototype_logit_mean",
        "window_top2__prototype_mean", "window_top2__prototype_top2",
        "window_mean__prototype_mean",
    ]
    result = {name: np.zeros((len(texts), config.NUM_FACTORS), dtype=np.float32)
              for name in names}
    batch_size = max(1, int(config.FACTOR_CROSS_ENCODER_BATCH_SIZE))
    for label, prototypes in enumerate(tqdm(FACTOR_PROTOTYPES, desc="pooling labels")):
        window_max = np.zeros((len(texts), len(prototypes)), dtype=np.float32)
        window_top2 = np.zeros_like(window_max)
        window_mean = np.zeros_like(window_max)
        for prototype_index, hypothesis in enumerate(prototypes):
            for start in range(0, len(texts), batch_size):
                batch = texts[start:start + batch_size]
                encoded = tokenizer(
                    batch, [hypothesis] * len(batch), padding=True,
                    truncation="only_first", max_length=config.FACTOR_NLI_MAX_LENGTH,
                    stride=min(128, config.FACTOR_NLI_MAX_LENGTH // 3),
                    return_overflowing_tokens=True, return_tensors="pt",
                )
                mapping = encoded.pop("overflow_to_sample_mapping").cpu().numpy()
                selected = []
                for local in range(len(batch)):
                    indices = np.flatnonzero(mapping == local)
                    if len(indices) > config.FACTOR_NLI_MAX_CHUNKS:
                        positions = np.linspace(
                            0, len(indices) - 1, config.FACTOR_NLI_MAX_CHUNKS
                        ).round().astype(int)
                        indices = indices[positions]
                    selected.extend(indices.tolist())
                selected = np.asarray(selected, dtype=np.int64)
                selected_mapping = mapping[selected]
                selected_tensor = torch.tensor(selected, dtype=torch.long)
                encoded = {
                    key: value.index_select(0, selected_tensor).to(device)
                    for key, value in encoded.items()
                }
                with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    logits = model(**encoded).logits.float()
                chunk_scores = torch.softmax(logits, dim=-1)[:, entailment].cpu().numpy()
                for local in range(len(batch)):
                    values = chunk_scores[selected_mapping == local]
                    row = start + local
                    window_max[row, prototype_index] = float(values.max())
                    window_mean[row, prototype_index] = float(values.mean())
                    window_top2[row, prototype_index] = float(
                        np.sort(values)[-min(2, len(values)):].mean()
                    )
        for prototype_mode in ("mean", "top2", "max", "logit_mean"):
            result[f"window_max__prototype_{prototype_mode}"][:, label] = _prototype_pool(
                window_max, prototype_mode
            )
        for prototype_mode in ("mean", "top2"):
            result[f"window_top2__prototype_{prototype_mode}"][:, label] = _prototype_pool(
                window_top2, prototype_mode
            )
        result["window_mean__prototype_mean"][:, label] = window_mean.mean(1)
    return result


def _macro(targets, probabilities, prevalence, ratio):
    return float(f1_score(
        targets, _rank_decode(probabilities, prevalence, ratio),
        average="macro", zero_division=0,
    ))


def evaluate_fold0(force=False):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = load_train_data().reset_index(drop=True)
    targets = np.vstack(frame.factor_vector.to_numpy()).astype(np.int8)
    folds = list(StratifiedGroupKFold(
        n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED
    ).split(np.zeros(len(frame)), frame.risk_label, frame.anon_user_id))
    train_idx, valid_idx = folds[0]
    checkpoint = V3_DIR / "fold0_model.pt"
    cache = OUTPUT_DIR / "fold0_variants.npz"
    variants = None
    if cache.exists() and not force:
        saved = np.load(cache)
        if (np.array_equal(saved["valid_indices"], valid_idx)
                and int(saved["checkpoint_signature"]) == checkpoint.stat().st_mtime_ns):
            variants = {key: saved[key] for key in saved.files
                        if key not in ("valid_indices", "checkpoint_signature")}
    if variants is None:
        tokenizer = AutoTokenizer.from_pretrained(
            config.FACTOR_NLI_MODEL_NAME, use_fast=True, local_files_only=True
        )
        device = torch.device(config.DEVICE)
        model = AutoModelForSequenceClassification.from_pretrained(
            config.FACTOR_NLI_MODEL_NAME, dtype=torch.float32, local_files_only=True
        ).to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        variants = _predict_variants(
            model, tokenizer, frame.text.iloc[valid_idx].tolist(), device
        )
        np.savez_compressed(
            cache, **variants, valid_indices=valid_idx,
            checkpoint_signature=checkpoint.stat().st_mtime_ns,
        )
        del model
        if device.type == "cuda": torch.cuda.empty_cache()

    base_saved = np.load(config.OUTPUT_DIR / "factor_cv" / "factor_oof_predictions.npz")
    base = (
        config.FACTOR_SEMANTIC_MODEL_WEIGHT * base_saved["semantic"]
        + config.FACTOR_CPU_ENSEMBLE_WEIGHT * base_saved["cpu"]
    )
    old_cross = np.load(
        config.OUTPUT_DIR / "factor_cross_encoder" / "oof_predictions.npz"
    )["probabilities"]
    calibration = json.loads(
        (V3_DIR / "calibration.json").read_text(encoding="utf-8")
    )
    saved_current = np.load(V3_DIR / "fold0_valid.npz")["probabilities"]
    reproduction_mae = float(np.abs(
        variants["window_max__prototype_mean"] - saved_current
    ).mean())
    prevalence = targets[train_idx].mean(0)
    rows = []
    for name, new_probability in variants.items():
        mixed = (
            float(calibration["base_weight"]) * base[valid_idx]
            + float(calibration["old_cross_weight"]) * old_cross[valid_idx]
            + float(calibration["new_cross_weight"]) * new_probability
        )
        rows.append({
            "variant": name,
            "fixed_macro_f1": _macro(
                targets[valid_idx], mixed, prevalence,
                float(calibration["prevalence_ratio"]),
            ),
            "standalone_macro_f1": _macro(
                targets[valid_idx], new_probability, prevalence, 1.0
            ),
        })
    rows.sort(key=lambda item: item["fixed_macro_f1"], reverse=True)
    baseline = next(x for x in rows if x["variant"] == "window_max__prototype_mean")
    payload = {
        "reproduction_mae": reproduction_mae, "baseline": baseline,
        "best": rows[0],
        "fixed_delta": rows[0]["fixed_macro_f1"] - baseline["fixed_macro_f1"],
        "variants": rows,
    }
    (OUTPUT_DIR / "fold0_results.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2)); return payload


if __name__ == "__main__":
    evaluate_fold0()
