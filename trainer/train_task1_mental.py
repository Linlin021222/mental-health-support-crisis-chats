"""Strict/full training for the MentalRoBERTa Task 1 risk expert."""
from __future__ import annotations

import json

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from configs.config import config
from datasets.factor_cache_builder import build_factor_cache
from models.task1_mental_model import (
    MentalRobertaRiskModel, optimizer_parameters, ordinal_class_probabilities,
)
from trainer.factor_train import FactorDataset
from utils.seed import seed_everything


OUTPUT = config.OUTPUT_DIR / "task1_mental"
STRICT_CHECKPOINT = OUTPUT / "strict_model.pt"
STRICT_PREDICTIONS = OUTPUT / "strict_predictions.npz"
STRICT_RESULTS = OUTPUT / "strict_results.json"
FULL_CHECKPOINT = OUTPUT / "full_model.pt"
TRAINING_VERSION = "mentalroberta-risk-v1"


def _collate(rows):
    return {
        "input_ids": torch.stack([row["input_ids"] for row in rows]),
        "attention_mask": torch.stack([row["attention_mask"] for row in rows]),
        "risk_labels": torch.tensor([int(row["risk_label"]) for row in rows], dtype=torch.long),
    }


def _loader(dataset, indices, shuffle):
    return DataLoader(
        Subset(dataset, indices), batch_size=config.BATCH_SIZE, shuffle=shuffle,
        collate_fn=_collate, num_workers=config.NUM_WORKERS,
        pin_memory=config.DEVICE == "cuda",
    )


class MentalRiskLoss(torch.nn.Module):
    def __init__(self, class_weights, ordinal_weights):
        super().__init__()
        self.register_buffer("class_weights", class_weights)
        self.register_buffer("ordinal_weights", ordinal_weights)

    def forward(self, output, labels):
        categorical = torch.nn.functional.cross_entropy(
            output["risk_logits"], labels, weight=self.class_weights,
            label_smoothing=0.03,
        )
        thresholds = torch.arange(config.NUM_RISK_CLASSES - 1, device=labels.device)
        ordinal_targets = (labels.unsqueeze(1) > thresholds).float()
        ordinal = torch.nn.functional.binary_cross_entropy_with_logits(
            output["ordinal_logits"], ordinal_targets,
            pos_weight=self.ordinal_weights,
        )
        return categorical + 0.30 * ordinal


def _loss(dataset, indices, device):
    labels = np.asarray([int(dataset.data[int(i)]["risk_label"]) for i in indices])
    counts = np.bincount(labels, minlength=config.NUM_RISK_CLASSES)
    class_weights = np.sqrt(len(indices) / np.maximum(counts, 1))
    class_weights = torch.tensor(
        class_weights / class_weights.mean(), dtype=torch.float, device=device,
    )
    ordinal = labels[:, None] > np.arange(config.NUM_RISK_CLASSES - 1)[None, :]
    positives = ordinal.sum(0)
    ordinal_weights = torch.tensor(
        np.sqrt((len(indices) - positives) / np.maximum(positives, 1)),
        dtype=torch.float, device=device,
    )
    return MentalRiskLoss(class_weights, ordinal_weights).to(device)


def _train_epoch(model, loader, criterion, optimizer, scheduler, scaler, device, epoch):
    model.train(); optimizer.zero_grad(set_to_none=True); losses = []
    for step, batch in enumerate(tqdm(loader, desc=f"mental-risk epoch {epoch}"), 1):
        ids = batch["input_ids"].to(device); mask = batch["attention_mask"].to(device)
        labels = batch["risk_labels"].to(device)
        with torch.autocast(device_type=device.type, enabled=config.FP16):
            loss = criterion(model(ids, mask), labels) / config.GRADIENT_ACCUMULATION
        scaler.scale(loss).backward()
        losses.append(float(loss.detach()) * config.GRADIENT_ACCUMULATION)
        if step % config.GRADIENT_ACCUMULATION == 0 or step == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
            old_scale = scaler.get_scale(); scaler.step(optimizer); scaler.update()
            if scaler.get_scale() >= old_scale:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)
    return float(np.mean(losses))


@torch.no_grad()
def _predict(model, loader, device):
    model.eval(); standard = []; ordinal = []; targets = []
    for batch in loader:
        output = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        standard.append(torch.softmax(output["risk_logits"], -1).cpu().numpy())
        ordinal.append(ordinal_class_probabilities(output["ordinal_logits"]).cpu().numpy())
        targets.append(batch["risk_labels"].numpy())
    return np.vstack(standard), np.vstack(ordinal), np.concatenate(targets)


def _metric(standard, ordinal, targets):
    rows = []
    for weight in (0.0, 0.10, 0.20, 0.30, 0.40, 0.50):
        probability = (1.0 - weight) * standard + weight * ordinal
        prediction = probability.argmax(1)
        rows.append({
            "ordinal_weight": weight,
            "risk_f1": float(f1_score(targets, prediction, average="weighted", zero_division=0)),
            "confusion": confusion_matrix(
                targets, prediction, labels=np.arange(config.NUM_RISK_CLASSES)
            ).tolist(),
        })
    rows.sort(key=lambda row: row["risk_f1"], reverse=True)
    return rows


def train_task1_mental_strict():
    seed_everything(config.SEED + 701)
    cache = build_factor_cache(train=True)
    dataset = FactorDataset(cache)
    labels = np.asarray([int(row["risk_label"]) for row in dataset.data])
    groups = np.asarray([str(row["anon_user_id"]) for row in dataset.data])
    train_idx, valid_idx = next(StratifiedGroupKFold(
        n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED
    ).split(np.zeros(len(dataset)), labels, groups))
    device = torch.device(config.DEVICE)
    train_loader = _loader(dataset, train_idx, True)
    valid_loader = _loader(dataset, valid_idx, False)
    model = MentalRobertaRiskModel(initialise_labels=True).to(device)
    criterion = _loss(dataset, train_idx, device)
    optimizer = AdamW(optimizer_parameters(model), weight_decay=config.WEIGHT_DECAY)
    updates = int(np.ceil(len(train_loader) / config.GRADIENT_ACCUMULATION)) * config.EPOCHS
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, max(1, int(updates * config.WARMUP_RATIO)), updates
    )
    scaler = torch.amp.GradScaler("cuda", enabled=config.FP16)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    best = {"risk_f1": -1.0}; best_arrays = None
    for epoch in range(1, config.EPOCHS + 1):
        loss = _train_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler, device, epoch
        )
        standard, ordinal, targets = _predict(model, valid_loader, device)
        grid = _metric(standard, ordinal, targets)
        print(
            f"mental-risk epoch={epoch} loss={loss:.4f} "
            f"risk_f1={grid[0]['risk_f1']:.4f} ordinal_weight={grid[0]['ordinal_weight']:.2f}"
        )
        if grid[0]["risk_f1"] > best["risk_f1"]:
            best = {"epoch": epoch, "train_loss": loss, **grid[0]}
            best_arrays = (standard.copy(), ordinal.copy(), targets.copy())
            torch.save(model.state_dict(), STRICT_CHECKPOINT)
    standard, ordinal, targets = best_arrays
    np.savez_compressed(
        STRICT_PREDICTIONS, valid_indices=valid_idx, standard=standard,
        ordinal=ordinal, targets=targets,
    )
    payload = {
        "training_version": TRAINING_VERSION,
        "train_posts": int(len(train_idx)), "valid_posts": int(len(valid_idx)),
        "train_users": int(len(np.unique(groups[train_idx]))),
        "valid_users": int(len(np.unique(groups[valid_idx]))),
        "overlapping_users": int(len(set(groups[train_idx]) & set(groups[valid_idx]))),
        "best": best,
    }
    STRICT_RESULTS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2)); return payload


def train_task1_mental_full():
    if not STRICT_RESULTS.exists():
        raise FileNotFoundError("Run MentalRoBERTa strict training before full training")
    seed_everything(config.SEED + 701)
    cache = build_factor_cache(train=True); dataset = FactorDataset(cache)
    indices = np.arange(len(dataset)); device = torch.device(config.DEVICE)
    loader = _loader(dataset, indices, True)
    model = MentalRobertaRiskModel(initialise_labels=True).to(device)
    criterion = _loss(dataset, indices, device)
    optimizer = AdamW(optimizer_parameters(model), weight_decay=config.WEIGHT_DECAY)
    selected = json.loads(STRICT_RESULTS.read_text(encoding="utf-8"))["best"]
    epochs = int(selected["epoch"])
    updates = int(np.ceil(len(loader) / config.GRADIENT_ACCUMULATION)) * epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, max(1, int(updates * config.WARMUP_RATIO)), updates
    )
    scaler = torch.amp.GradScaler("cuda", enabled=config.FP16)
    for epoch in range(1, epochs + 1):
        loss = _train_epoch(model, loader, criterion, optimizer, scheduler, scaler, device, epoch)
        print(f"mental-risk-full epoch={epoch} loss={loss:.4f}")
    torch.save(model.state_dict(), FULL_CHECKPOINT)
    print(f"MentalRoBERTa full risk checkpoint: {FULL_CHECKPOINT}")
    return FULL_CHECKPOINT


def evaluate_task1_mental_strict_checkpoint(epoch=1):
    """Evaluate an already saved strict checkpoint without retraining it."""
    if not STRICT_CHECKPOINT.exists():
        raise FileNotFoundError(f"Missing checkpoint: {STRICT_CHECKPOINT}")
    seed_everything(config.SEED + 701)
    cache = build_factor_cache(train=True)
    dataset = FactorDataset(cache)
    labels = np.asarray([int(row["risk_label"]) for row in dataset.data])
    groups = np.asarray([str(row["anon_user_id"]) for row in dataset.data])
    train_idx, valid_idx = next(StratifiedGroupKFold(
        n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED
    ).split(np.zeros(len(dataset)), labels, groups))
    device = torch.device(config.DEVICE)
    model = MentalRobertaRiskModel(initialise_labels=False).to(device)
    model.load_state_dict(torch.load(STRICT_CHECKPOINT, map_location=device))
    standard, ordinal, targets = _predict(
        model, _loader(dataset, valid_idx, False), device
    )
    grid = _metric(standard, ordinal, targets)
    best = {"epoch": int(epoch), "train_loss": None, **grid[0]}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        STRICT_PREDICTIONS, valid_indices=valid_idx, standard=standard,
        ordinal=ordinal, targets=targets,
    )
    payload = {
        "training_version": TRAINING_VERSION,
        "train_posts": int(len(train_idx)), "valid_posts": int(len(valid_idx)),
        "train_users": int(len(np.unique(groups[train_idx]))),
        "valid_users": int(len(np.unique(groups[valid_idx]))),
        "overlapping_users": int(len(set(groups[train_idx]) & set(groups[valid_idx]))),
        "best": best,
    }
    STRICT_RESULTS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return payload


if __name__ == "__main__":
    train_task1_mental_strict()
