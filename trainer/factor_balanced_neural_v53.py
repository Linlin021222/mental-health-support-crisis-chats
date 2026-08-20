"""Factor-balanced MentalRoBERTa with implemented tail sampling (V53).

The original configuration exposed FACTOR_TAIL_SAMPLING_ALPHA, but the
standalone DataLoader never consumed it.  V53 uses the factor-balanced user
folds and a capped weighted sampler.  Meaning-in-life examples receive a
moderate additional boost, and repeated annotations only weight an auxiliary
meaning loss; the benchmark target remains binary.
"""
from __future__ import annotations

import json

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from tqdm import tqdm

from configs.config import config
from datasets.factor_cache_builder import build_factor_cache
from inference.factor_boundary_lexicon_v50 import boundary_flags
from inference.factor_nli import _rank_decode
from models.factor_model import factor_optimizer_parameters
from preprocess.preprocess import load_train_data
from trainer.factor_rare_semantic_v49 import _baseline_probability
from trainer.factor_train import FactorDataset, WeightedGroupedASL, _collate, _loss_and_model, _probabilities
from utils.multilabel_group_split import multilabel_group_folds, split_audit
from utils.seed import seed_everything


OUTPUT = config.OUTPUT_DIR / "factor_balanced_neural_v53"
RESULTS = OUTPUT / "fold0_results.json"
LABEL = 23
EPOCHS = 3
MEANING_AUX_WEIGHT = .08
MEANING_SAMPLE_BOOST = 1.75
TRAINING_VERSION = "factor-balanced-mentalroberta-tail-v53"


def _sampler_loader(dataset, indices, targets, seed):
    support = targets[indices].sum(0).clip(min=1)
    tail = np.sqrt(len(indices) / support)
    tail = tail / max(1e-6, float(np.median(tail)))
    row_tail = (targets[indices] * tail[None, :]).max(1)
    weights = 1.0 + config.FACTOR_TAIL_SAMPLING_ALPHA * np.clip(row_tail - 1.0, 0.0, 2.0)
    weights += MEANING_SAMPLE_BOOST * targets[indices, LABEL]
    weights = np.clip(weights, 1.0, 4.0)
    generator = torch.Generator().manual_seed(seed)
    sampler = WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double), num_samples=len(indices),
        replacement=True, generator=generator,
    )
    loader = DataLoader(
        Subset(dataset, indices), batch_size=config.BATCH_SIZE, sampler=sampler,
        collate_fn=_collate, num_workers=0, pin_memory=config.DEVICE == "cuda",
    )
    return loader, {
        "mean_sample_weight": float(weights.mean()),
        "max_sample_weight": float(weights.max()),
        "meaning_sample_weight": float(weights[targets[indices, LABEL] == 1].mean()),
        "expected_meaning_draws_per_epoch": float(
            len(indices) * weights[targets[indices, LABEL] == 1].sum() / weights.sum()
        ),
    }


def _train_epoch(model, loader, loss_fn, optimizer, scaler, device, epoch, pos_weight):
    model.train(); optimizer.zero_grad(set_to_none=True); losses = []
    for step, batch in enumerate(tqdm(loader, desc=f"V53 factor epoch {epoch}"), 1):
        ids = batch["input_ids"].to(device); mask = batch["attention_mask"].to(device)
        targets = batch["factor_vectors"].to(device)
        counts = batch["factor_counts"].to(device)
        with torch.autocast(device_type=device.type, enabled=config.FP16):
            logits, semantic = model(ids, mask, return_semantic=True)
            classification = loss_fn(logits, targets, counts)
            semantic_loss = loss_fn(semantic, targets)
            meaning_raw = torch.nn.functional.binary_cross_entropy_with_logits(
                logits[:, LABEL], targets[:, LABEL], pos_weight=pos_weight,
                reduction="none",
            )
            repeat = 1.0 + .25 * torch.log1p((counts[:, LABEL] - 1.0).clamp_min(0.0))
            meaning_loss = (meaning_raw * torch.where(
                targets[:, LABEL] > 0, repeat, torch.ones_like(repeat)
            )).mean()
            loss = (classification + config.FACTOR_SEMANTIC_LOSS_WEIGHT * semantic_loss
                    + MEANING_AUX_WEIGHT * meaning_loss)
            loss = loss / config.GRADIENT_ACCUMULATION
        scaler.scale(loss).backward(); losses.append(float(loss.detach()) * config.GRADIENT_ACCUMULATION)
        if step % config.GRADIENT_ACCUMULATION == 0 or step == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
            scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
    return float(np.mean(losses))


def _rank(values):
    order = np.argsort(np.argsort(values, kind="mergesort"), kind="mergesort")
    return (order + .5) / len(values)


def train_fold0():
    OUTPUT.mkdir(parents=True, exist_ok=True); seed_everything(config.SEED + 5300)
    cache = build_factor_cache(train=True)
    dataset = FactorDataset(cache)
    frame = load_train_data().reset_index(drop=True)
    targets = np.vstack(frame.factor_vector.to_numpy()).astype(np.int8)
    groups = frame.anon_user_id.astype(str).to_numpy()
    risk = frame.risk_label.to_numpy(dtype=np.int64)
    folds = multilabel_group_folds(targets, groups, risk, 5, config.SEED + 47)
    train_idx, valid_idx = folds[0]
    train_loader, sampler_audit = _sampler_loader(
        dataset, train_idx, targets, config.SEED + 5300,
    )
    valid_loader = DataLoader(
        Subset(dataset, valid_idx), batch_size=config.BATCH_SIZE, shuffle=False,
        collate_fn=_collate, num_workers=0, pin_memory=config.DEVICE == "cuda",
    )
    device = torch.device(config.DEVICE)
    model, loss_fn = _loss_and_model(dataset, train_idx, device)
    if not isinstance(loss_fn, WeightedGroupedASL):
        raise RuntimeError("V53 requires the accepted weighted ASL loss")
    optimizer = AdamW(factor_optimizer_parameters(model), weight_decay=config.WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=config.FP16)
    positive = max(1, int(targets[train_idx, LABEL].sum()))
    pos_weight = torch.tensor(
        min(12.0, float(np.sqrt((len(train_idx)-positive) / positive))),
        device=device,
    )
    histories, epochs = [], []
    for epoch in range(1, EPOCHS + 1):
        loss = _train_epoch(
            model, train_loader, loss_fn, optimizer, scaler, device, epoch, pos_weight,
        )
        probability, truth = _probabilities(model, valid_loader, device)
        epochs.append(probability); histories.append(loss)
        print(f"V53 fold=0 epoch={epoch} loss={loss:.5f}", flush=True)

    # Epoch 3 is predeclared from the established Task-2 learning curve; it is
    # not selected on this validation fold.
    probability = epochs[-1]
    torch.save(model.state_dict(), OUTPUT / "fold0_model.pt")
    np.savez_compressed(OUTPUT / "fold0_valid.npz", probabilities=probability,
                        valid_indices=valid_idx, histories=np.asarray(histories))
    prevalence = targets[train_idx].mean(0)
    old = _baseline_probability()[valid_idx]
    flags = boundary_flags(frame.text.iloc[valid_idx].astype(str).tolist())
    old = old.copy(); old[:, 13] += .10 * flags[:, 13]
    old[:, 18] += .50 * flags[:, 18]; old[:, 23] += .20 * flags[:, 23]
    baseline_prediction = _rank_decode(old, prevalence, 1.10)
    candidates = []
    for weight in (.25, .50, .75, 1.0):
        mixed = old.copy()
        for label in range(config.NUM_FACTORS):
            mixed[:, label] = ((1-weight) * _rank(old[:, label])
                               + weight * _rank(probability[:, label]))
        prediction = _rank_decode(mixed, prevalence, 1.10)
        candidates.append({
            "semantic_replacement_weight": weight,
            "macro_f1": float(f1_score(truth, prediction, average="macro", zero_division=0)),
            "meaning_f1": float(f1_score(truth[:, LABEL], prediction[:, LABEL], zero_division=0)),
            "meaning_roc_auc": float(roc_auc_score(truth[:, LABEL], probability[:, LABEL])),
            "meaning_pr_auc": float(average_precision_score(truth[:, LABEL], probability[:, LABEL])),
        })
    payload = {
        "training_version": TRAINING_VERSION,
        "evaluation": "untouched factor-balanced user-disjoint fold0; fixed epoch 3",
        "split_audit": split_audit([folds[0]], targets, groups),
        "sampler_audit": sampler_audit,
        "history": histories,
        "baseline_macro_f1": float(f1_score(
            truth, baseline_prediction, average="macro", zero_division=0)),
        "baseline_meaning_f1": float(f1_score(
            truth[:, LABEL], baseline_prediction[:, LABEL], zero_division=0)),
        "candidates": candidates,
        "adopted": False,
        "note": "Run five folds only if fold0 improves both macro F1 and meaning F1.",
    }
    RESULTS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    return payload


if __name__ == "__main__":
    train_fold0()
