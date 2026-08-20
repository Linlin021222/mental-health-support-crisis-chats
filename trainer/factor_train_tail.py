"""Tail-aware strict Task 2 experiment with contextual label prototypes."""
import json
import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from tqdm import tqdm

from configs.config import config
from datasets.factor_cache_builder import build_factor_cache
from models.factor_model import factor_optimizer_parameters
from trainer.factor_train import FactorDataset, _collate, _loss_and_model, _probabilities
from utils.factor_calibration import calibrate_factor_thresholds, apply_prior_topk


CHECKPOINT = config.OUTPUT_DIR / "factor_strict_model_tail.pt"
RESULTS = config.OUTPUT_DIR / "factor_strict_tail_results.json"


def _balanced_loader(dataset, indices):
    targets = np.vstack([dataset.data[int(i)]["factor_vector"].numpy() for i in indices])
    frequency = targets.sum(0).clip(min=1.0)
    rarity = np.sqrt(len(indices) / frequency)
    positive_count = targets.sum(1)
    sample_rarity = (targets * rarity[None, :]).sum(1) / np.maximum(positive_count, 1.0)
    sample_rarity[positive_count == 0] = 1.0
    sample_rarity = sample_rarity / sample_rarity.mean()
    alpha = float(config.FACTOR_TAIL_SAMPLING_ALPHA)
    weights = np.clip((1.0 - alpha) + alpha * sample_rarity, 0.5, 4.0)
    subset = Subset(dataset, indices)
    sampler = WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double), len(indices), replacement=True,
        generator=torch.Generator().manual_seed(config.SEED),
    )
    print(
        f"tail sampler weights min/mean/max={weights.min():.3f}/"
        f"{weights.mean():.3f}/{weights.max():.3f}"
    )
    return DataLoader(
        subset, batch_size=config.BATCH_SIZE, sampler=sampler, collate_fn=_collate,
        num_workers=config.NUM_WORKERS, pin_memory=config.DEVICE == "cuda",
    )


def _plain_loader(dataset, indices):
    return DataLoader(
        Subset(dataset, indices), batch_size=config.BATCH_SIZE, shuffle=False,
        collate_fn=_collate, num_workers=config.NUM_WORKERS,
    )


def _ranking_loss(semantic_logits, targets, margin=0.20):
    """Force every positive label above the hardest negative for each post."""
    losses = []
    for scores, labels in zip(semantic_logits, targets.bool()):
        if not labels.any() or labels.all():
            continue
        hardest_negative = scores[~labels].max()
        losses.append(torch.nn.functional.softplus(margin + hardest_negative - scores[labels]).mean())
    return torch.stack(losses).mean() if losses else semantic_logits.sum() * 0.0


def _train_epoch(model, loader, loss_fn, optimizer, scaler, device, epoch):
    model.train(); optimizer.zero_grad(set_to_none=True); losses = []
    for step, batch in enumerate(tqdm(loader, desc=f"factor-tail epoch {epoch}"), 1):
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        targets = batch["factor_vectors"].to(device)
        with torch.autocast(device_type=device.type, enabled=config.FP16):
            logits, semantic_logits = model(ids, mask, return_semantic=True)
            classification = loss_fn(logits, targets)
            semantic = loss_fn(semantic_logits, targets)
            ranking = _ranking_loss(semantic_logits, targets)
            loss = (
                classification + config.FACTOR_SEMANTIC_LOSS_WEIGHT * semantic
                + config.FACTOR_RANKING_LOSS_WEIGHT * ranking
            ) / config.GRADIENT_ACCUMULATION
        scaler.scale(loss).backward()
        losses.append(float(loss.detach()) * config.GRADIENT_ACCUMULATION)
        if step % config.GRADIENT_ACCUMULATION == 0 or step == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
            scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
    return float(np.mean(losses))


def train_factor_tail_strict():
    cache = build_factor_cache(train=True)
    dataset = FactorDataset(cache)
    risk = np.asarray([x["risk_label"] for x in dataset.data])
    groups = np.asarray([x["anon_user_id"] for x in dataset.data])
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=config.SEED)
    outer_train, valid_idx = next(outer.split(np.zeros(len(dataset)), risk, groups))
    inner = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=config.SEED + 91)
    fit_rel, cal_rel = next(inner.split(outer_train, groups=groups[outer_train]))
    fit_idx, cal_idx = outer_train[fit_rel], outer_train[cal_rel]
    device = torch.device(config.DEVICE)
    model, loss_fn = _loss_and_model(dataset, fit_idx, device)
    optimizer = AdamW(factor_optimizer_parameters(model), weight_decay=config.WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=config.FP16)
    fit_loader = _balanced_loader(dataset, fit_idx)
    cal_loader, valid_loader = _plain_loader(dataset, cal_idx), _plain_loader(dataset, valid_idx)
    fit_targets = np.vstack([dataset.data[int(i)]["factor_vector"].numpy() for i in fit_idx])
    prevalence = fit_targets.mean(0)
    best = {"macro_f1": -1.0}
    history = []
    for epoch in range(1, config.FACTOR_EPOCHS + 1):
        loss = _train_epoch(model, fit_loader, loss_fn, optimizer, scaler, device, epoch)
        cal_probability, cal_targets = _probabilities(model, cal_loader, device)
        thresholds = calibrate_factor_thresholds(
            cal_targets, cal_probability, reference_prevalence=prevalence
        )
        valid_probability, valid_targets = _probabilities(model, valid_loader, device)
        threshold_pred = valid_probability >= thresholds[None, :]
        threshold_score = f1_score(valid_targets, threshold_pred, average="macro", zero_division=0)
        topk_scores = {}
        for ratio in (0.9, 1.0, 1.1, 1.25):
            pred = apply_prior_topk(valid_probability, prevalence, ratio)
            topk_scores[str(ratio)] = f1_score(
                valid_targets, pred, average="macro", zero_division=0
            )
        score = max([threshold_score] + list(topk_scores.values()))
        selected = {
            "epoch": epoch, "train_loss": loss, "threshold_f1": threshold_score,
            "topk_f1": topk_scores, "macro_f1": score,
        }
        history.append(selected); print("factor-tail", selected)
        if score > best["macro_f1"]:
            best = selected
            torch.save(model.state_dict(), CHECKPOINT)
            np.savez_compressed(
                config.OUTPUT_DIR / "factor_tail_strict_probabilities.npz",
                valid_probability=valid_probability, valid_targets=valid_targets,
                cal_probability=cal_probability, cal_targets=cal_targets,
                valid_indices=valid_idx, cal_indices=cal_idx, fit_indices=fit_idx,
            )
            RESULTS.write_text(json.dumps({"best": best, "history": history}, indent=2), encoding="utf-8")
    print("Best factor-tail strict", best)
    return best


if __name__ == "__main__":
    train_factor_tail_strict()
