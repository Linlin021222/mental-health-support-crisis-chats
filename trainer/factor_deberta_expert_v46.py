"""Strict user-disjoint DeBERTa factor expert and fixed-rank blend test."""
from __future__ import annotations

import json

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from configs.config import config
from datasets.cache_builder import build_cache
from datasets.dataset import SuicideRiskDataset
from inference.factor_nli import _rank_decode
from models.factor_deberta_expert_v46 import DebertaFactorExpert
from preprocess.preprocess import load_train_data
from trainer.factor_mhlat_v4 import _current_components
from trainer.factor_sentence_evidence_cv_v27 import _user_bootstrap
from trainer.factor_train import WeightedGroupedASL
from utils.seed import seed_everything


OUTPUT = config.OUTPUT_DIR / "factor_deberta_expert_v46"
RESULTS = OUTPUT / "fold0_results.json"
CHECKPOINT = OUTPUT / "fold0_model.pt"
PREDICTIONS = OUTPUT / "fold0_valid.npz"
TRAINING_VERSION = "deberta-v3-semantic-factor-expert-v46"
EPOCHS = 2
FIXED_WEIGHT = .10


def _collate(rows):
    return {
        "input_ids": torch.stack([row["input_ids"] for row in rows]),
        "attention_mask": torch.stack([row["attention_mask"] for row in rows]),
        "targets": torch.stack([row["factor_vector"] for row in rows]),
    }


def _loader(dataset, indices, shuffle):
    return DataLoader(
        Subset(dataset, indices), batch_size=1, shuffle=shuffle,
        collate_fn=_collate, num_workers=0, pin_memory=True,
    )


def _rank_columns(probability):
    result = np.zeros_like(probability, dtype=np.float32)
    for label in range(probability.shape[1]):
        order = np.argsort(probability[:, label], kind="mergesort")
        result[order, label] = np.linspace(0., 1., len(probability), dtype=np.float32)
    return result


def _train(model, loader, loss_fn, device):
    backbone = [p for name, p in model.named_parameters() if name.startswith("encoder.")]
    heads = [p for name, p in model.named_parameters() if not name.startswith("encoder.")]
    optimizer = AdamW([
        {"params": backbone, "lr": 5e-6}, {"params": heads, "lr": 3e-5},
    ], weight_decay=config.WEIGHT_DECAY)
    accumulation = config.GRADIENT_ACCUMULATION
    updates = EPOCHS * max(1, int(np.ceil(len(loader) / accumulation)))
    scheduler = get_cosine_schedule_with_warmup(optimizer, max(1, int(.08 * updates)), updates)
    scaler = torch.amp.GradScaler("cuda", enabled=config.FP16); history = []
    for epoch in range(1, EPOCHS + 1):
        model.train(); optimizer.zero_grad(set_to_none=True); losses = []
        for step, batch in enumerate(tqdm(loader, desc=f"V46 DeBERTa epoch {epoch}/{EPOCHS}"), 1):
            targets = batch["targets"].to(device)
            with torch.autocast(device_type=device.type, enabled=config.FP16):
                logits, semantic = model(
                    batch["input_ids"].to(device), batch["attention_mask"].to(device),
                    return_semantic=True,
                )
                loss = (loss_fn(logits, targets) + .08 * loss_fn(semantic, targets)) / accumulation
            scaler.scale(loss).backward(); losses.append(float(loss.detach()) * accumulation)
            if step % accumulation == 0 or step == len(loader):
                scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                old = scaler.get_scale(); scaler.step(optimizer); scaler.update()
                if scaler.get_scale() >= old: scheduler.step()
                optimizer.zero_grad(set_to_none=True)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses))}
        history.append(row); print(json.dumps(row), flush=True)
    return history


@torch.no_grad()
def _predict(model, loader, device):
    model.eval(); values = []
    for batch in tqdm(loader, desc="V46 strict validation"):
        with torch.autocast(device_type=device.type, enabled=config.FP16):
            logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        values.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.vstack(values).astype(np.float32)


def _mean_metric(truth, probability, metric):
    values = []
    for label in range(24):
        if np.unique(truth[:, label]).size < 2: continue
        values.append(metric(truth[:, label], probability[:, label]))
    return float(np.mean(values))


def main():
    if not torch.cuda.is_available(): raise RuntimeError("V46 requires CUDA")
    OUTPUT.mkdir(parents=True, exist_ok=True); seed_everything(config.SEED + 4646)
    cache = config.CACHE_DIR / "train_cache.pt"
    if not cache.exists(): build_cache(train=True)
    dataset = SuicideRiskDataset(cache); frame = load_train_data().reset_index(drop=True)
    current, targets, calibration = _current_components(); targets = targets.astype(np.float32)
    groups = frame.anon_user_id.astype(str).to_numpy(); risks = frame.risk_label.to_numpy()
    train_idx, valid_idx = next(StratifiedGroupKFold(
        5, shuffle=True, random_state=config.SEED,
    ).split(np.zeros(len(frame)), risks, groups))
    positives = torch.tensor(targets[train_idx].sum(0), dtype=torch.float32, device="cuda")
    weights = torch.sqrt((len(train_idx) - positives) / positives.clamp_min(1.)).clamp(1., 12.)
    model = DebertaFactorExpert().to("cuda")
    history = _train(model, _loader(dataset, train_idx, True), WeightedGroupedASL(weights).cuda(), torch.device("cuda"))
    candidate = _predict(model, _loader(dataset, valid_idx, False), torch.device("cuda"))
    torch.save(model.state_dict(), CHECKPOINT)
    np.savez_compressed(PREDICTIONS, valid_indices=valid_idx, probabilities=candidate,
                        training_version=TRAINING_VERSION)

    truth = targets[valid_idx].astype(np.int8); prevalence = targets[train_idx].mean(0)
    ratio = float(calibration["prevalence_ratio"])
    current_rank = _rank_columns(current[valid_idx]); candidate_rank = _rank_columns(candidate)
    baseline_prediction = _rank_decode(current_rank, prevalence, ratio)
    baseline = float(f1_score(truth, baseline_prediction, average="macro", zero_division=0))
    grid = []
    for weight in (0., .05, .10, .15, .20, .30, .50):
        mixed = (1. - weight) * current_rank + weight * candidate_rank
        prediction = _rank_decode(mixed, prevalence, ratio)
        grid.append({"weight": weight, "macro_f1": float(f1_score(
            truth, prediction, average="macro", zero_division=0,
        ))})
    fixed = next(row for row in grid if row["weight"] == FIXED_WEIGHT)
    fixed["delta"] = fixed["macro_f1"] - baseline
    fixed_prediction = _rank_decode(
        (1. - FIXED_WEIGHT) * current_rank + FIXED_WEIGHT * candidate_rank,
        prevalence, ratio,
    )
    bootstrap = _user_bootstrap(
        truth, baseline_prediction, fixed_prediction, groups[valid_idx],
        seed=464646, draws=3000,
    )
    standalone_prediction = _rank_decode(candidate_rank, prevalence, ratio)
    payload = {
        "training_version": TRAINING_VERSION,
        "evaluation_scope": "strict user-disjoint fold0; fixed two epochs and 10% rank blend",
        "history": history,
        "baseline_macro_f1": baseline,
        "standalone_macro_f1": float(f1_score(
            truth, standalone_prediction, average="macro", zero_division=0,
        )),
        "standalone_macro_roc_auc": _mean_metric(truth, candidate, roc_auc_score),
        "standalone_macro_pr_auc": _mean_metric(truth, candidate, average_precision_score),
        "fixed_10pct": fixed,
        "grid_diagnostic": sorted(grid, key=lambda row: row["macro_f1"], reverse=True),
        "user_cluster_bootstrap": bootstrap,
        "promising": bool(fixed["delta"] >= .003 and bootstrap["positive_fraction"] >= .70),
        "adopted": False,
    }
    RESULTS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True); return payload


if __name__ == "__main__": main()
