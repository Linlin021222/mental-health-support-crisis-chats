"""Second-seed prototype-MIL fold ablation for Task 2 ensemble diversity."""
from __future__ import annotations

import json

import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_cosine_schedule_with_warmup

from configs.config import config
from inference.factor_nli import _entailment_index, _rank_decode
from preprocess.preprocess import load_train_data
from trainer.factor_cross_encoder_v2 import (
    OLD_DIR, OUTPUT_DIR as V3_DIR, PrototypePairDataset, _collator,
    _predict, _prototype_pairs,
)
from utils.seed import seed_everything


TRAINING_VERSION = "prototype-mil-second-seed-v5"
OUTPUT_DIR = config.OUTPUT_DIR / "factor_seed_ensemble_v5"


def train_fold0():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = load_train_data().reset_index(drop=True)
    targets = np.vstack(frame.factor_vector.to_numpy()).astype(np.int8)
    counts = np.vstack(frame.factor_counts.to_numpy()).astype(np.float32)
    train_idx, valid_idx = next(StratifiedGroupKFold(
        n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED
    ).split(np.zeros(len(frame)), frame.risk_label, frame.anon_user_id))
    checkpoint = OUTPUT_DIR / "fold0_model.pt"
    prediction_file = OUTPUT_DIR / "fold0_valid.npz"
    probability = None; summary = None
    if checkpoint.exists() and prediction_file.exists():
        saved = np.load(prediction_file, allow_pickle=True)
        summary = json.loads(str(saved["summary"]))
        if (np.array_equal(saved["valid_indices"], valid_idx)
                and summary.get("training_version") == TRAINING_VERSION):
            probability = saved["probabilities"].astype(np.float32)
            print("Task 2 second seed fold 0: resumed")

    if probability is None:
        seed_everything(config.SEED + 1100)
        tokenizer = AutoTokenizer.from_pretrained(
            config.FACTOR_NLI_MODEL_NAME, use_fast=True, local_files_only=True
        )
        device = torch.device(config.DEVICE)
        model = AutoModelForSequenceClassification.from_pretrained(
            config.FACTOR_NLI_MODEL_NAME, dtype=torch.float32, local_files_only=True
        ).to(device)
        old_checkpoint = OLD_DIR / "fold0_model.pt"
        model.load_state_dict(torch.load(old_checkpoint, map_location=device))
        hard_file = V3_DIR / "fold0_train_hard.npz"
        hard_saved = np.load(hard_file)
        if not np.array_equal(hard_saved["train_indices"], train_idx):
            raise ValueError("Second-seed hard-negative indices do not match fold 0")
        pairs = _prototype_pairs(
            targets[train_idx], counts[train_idx], hard_saved["probabilities"],
            config.SEED + 1100,
        )
        loader = DataLoader(
            PrototypePairDataset(frame.text.iloc[train_idx].tolist(), pairs),
            batch_size=config.FACTOR_PROTOTYPE_TRAIN_BATCH_SIZE, shuffle=True,
            collate_fn=_collator(tokenizer), num_workers=0,
        )
        model.gradient_checkpointing_enable(); model.config.use_cache = False
        entailment = _entailment_index(model)
        optimizer = AdamW(model.parameters(), lr=3e-6, weight_decay=config.WEIGHT_DECAY)
        accumulation = config.FACTOR_PROTOTYPE_ACCUMULATION
        updates = int(np.ceil(len(loader) / accumulation))
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, max(1, int(updates * 0.08)), max(1, updates)
        )
        scaler = torch.amp.GradScaler("cuda", enabled=config.FP16)
        model.train(); optimizer.zero_grad(set_to_none=True); losses = []
        for step, batch in enumerate(tqdm(loader, desc="second-seed fold 0"), 1):
            binary = batch.pop("targets").to(device).float()
            weights = batch.pop("weights").to(device)
            pair_mapping = batch.pop("pair_mapping").to(device)
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.autocast(device_type=device.type, enabled=config.FP16):
                chunk_logits = model(**batch).logits
                margin = chunk_logits[:, entailment] - chunk_logits[:, 1 - entailment]
                pair_margin = torch.stack([
                    margin[pair_mapping == pair].max() for pair in range(len(binary))
                ])
                raw = torch.nn.functional.binary_cross_entropy_with_logits(
                    pair_margin, binary, reduction="none"
                )
                loss = (raw * weights).mean() / accumulation
            scaler.scale(loss).backward(); losses.append(float(loss.detach()) * accumulation)
            if step % accumulation == 0 or step == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
                old_scale = scaler.get_scale(); scaler.step(optimizer); scaler.update()
                if scaler.get_scale() >= old_scale: scheduler.step()
                optimizer.zero_grad(set_to_none=True)
        probability = _predict(
            model, tokenizer, frame.text.iloc[valid_idx].tolist(), device
        )
        summary = {
            "fold": 0, "training_version": TRAINING_VERSION,
            "seed": config.SEED + 1100, "pair_count": len(pairs),
            "train_loss": float(np.mean(losses)),
        }
        torch.save(model.state_dict(), checkpoint)
        np.savez_compressed(
            prediction_file, probabilities=probability, valid_indices=valid_idx,
            summary=json.dumps(summary),
        )
        del model
        if device.type == "cuda": torch.cuda.empty_cache()

    base_saved = np.load(config.OUTPUT_DIR / "factor_cv" / "factor_oof_predictions.npz")
    base = .7 * base_saved["semantic"][valid_idx] + .3 * base_saved["cpu"][valid_idx]
    old = np.load(
        config.OUTPUT_DIR / "factor_cross_encoder" / "oof_predictions.npz"
    )["probabilities"][valid_idx]
    first_seed = np.load(V3_DIR / "fold0_valid.npz")["probabilities"]
    calibration = json.loads((V3_DIR / "calibration.json").read_text(encoding="utf-8"))
    prevalence = targets[train_idx].mean(0)
    rows = []
    for second_weight in (0.0, 0.25, 0.50, 0.75, 1.0):
        prototype = (1.0 - second_weight) * first_seed + second_weight * probability
        mixed = (
            calibration["base_weight"] * base
            + calibration["old_cross_weight"] * old
            + calibration["new_cross_weight"] * prototype
        )
        rows.append({
            "second_seed_weight": second_weight,
            "fixed_macro_f1": float(f1_score(
                targets[valid_idx], _rank_decode(
                    mixed, prevalence, calibration["prevalence_ratio"]
                ), average="macro", zero_division=0,
            )),
            "prototype_macro_f1": float(f1_score(
                targets[valid_idx], _rank_decode(prototype, prevalence, 1.0),
                average="macro", zero_division=0,
            )),
        })
    rows.sort(key=lambda item: item["fixed_macro_f1"], reverse=True)
    baseline = next(x for x in rows if x["second_seed_weight"] == 0.0)
    result = {
        "training": summary, "baseline": baseline, "best": rows[0],
        "fixed_delta": rows[0]["fixed_macro_f1"] - baseline["fixed_macro_f1"],
        "grid": rows,
    }
    (OUTPUT_DIR / "fold0_results.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2)); return result


if __name__ == "__main__":
    train_fold0()
