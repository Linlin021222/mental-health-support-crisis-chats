"""Strict five-fold MHLAT/label-centre refinement for Task 2.

This experiment is intentionally isolated from the accepted prototype-MIL-v3
pipeline.  Each fold starts from its completed one-hop MentalRoBERTa fold and
is accepted only when nested user-disjoint model selection improves the fixed
current ensemble, not merely the optimistic full OOF score.
"""
from __future__ import annotations

import json

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.optim import AdamW
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from configs.config import config
from datasets.factor_cache_builder import build_factor_cache
from models.factor_mhlat_model import MentalRobertaMHLATModel, mhlat_optimizer_parameters
from preprocess.preprocess import load_train_data
from trainer.factor_train import FactorDataset, WeightedGroupedASL, _loader
from utils.factor_calibration import apply_prior_topk
from utils.seed import seed_everything


TRAINING_VERSION = "mhlat-label-centre-v4"
OUTPUT_DIR = config.OUTPUT_DIR / "factor_mhlat_v4"
OOF_FILE = OUTPUT_DIR / "oof_predictions.npz"
RESULT_FILE = OUTPUT_DIR / "cv_results.json"
CALIBRATION_FILE = OUTPUT_DIR / "calibration.json"
BASE_DIR = config.OUTPUT_DIR / "factor_cv"
BASE_OOF_FILE = BASE_DIR / "factor_oof_predictions.npz"
OLD_CROSS_OOF_FILE = config.OUTPUT_DIR / "factor_cross_encoder" / "oof_predictions.npz"
V3_OOF_FILE = config.OUTPUT_DIR / "factor_cross_encoder_v2" / "oof_predictions.npz"
V3_CALIBRATION_FILE = config.OUTPUT_DIR / "factor_cross_encoder_v2" / "calibration.json"


def _fold_paths(fold):
    return OUTPUT_DIR / f"fold{fold}_model.pt", OUTPUT_DIR / f"fold{fold}_valid.npz"


def _loss(dataset, indices, device):
    targets = torch.stack([dataset.data[int(i)]["factor_vector"] for i in indices]).float()
    positives = targets.sum(0)
    weights = torch.sqrt((len(indices) - positives) / positives.clamp_min(1.0))
    weights = weights.clamp(min=1.0, max=12.0).to(device)
    return WeightedGroupedASL(weights).to(device), (weights / weights.mean()).detach()


def _centre_loss(centre_logits, targets, class_weights):
    rows, labels = torch.where(targets > 0.5)
    if len(rows) == 0:
        return centre_logits.sum() * 0.0
    raw = F.cross_entropy(centre_logits[rows, labels], labels, reduction="none")
    # Square-root tail weights already used by ASL are normalised here so the
    # auxiliary objective does not silently change the total loss scale.
    return (raw * class_weights[labels]).mean()


def _train_epoch(model, loader, loss_fn, centre_weights, optimizer, scheduler,
                 scaler, device, epoch):
    model.train(); optimizer.zero_grad(set_to_none=True); losses = []
    for step, batch in enumerate(tqdm(loader, desc=f"MHLAT-v4 epoch {epoch}"), 1):
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        targets = batch["factor_vectors"].to(device)
        counts = batch["factor_counts"].to(device)
        with torch.autocast(device_type=device.type, enabled=config.FP16):
            logits, semantic_logits, centre_logits = model(ids, mask, return_aux=True)
            classification = loss_fn(logits, targets, counts)
            semantic = loss_fn(semantic_logits, targets)
            contrastive = _centre_loss(centre_logits, targets, centre_weights)
            loss = (
                classification
                + config.FACTOR_MHLAT_SEMANTIC_WEIGHT * semantic
                + config.FACTOR_MHLAT_CONTRASTIVE_WEIGHT * contrastive
            ) / config.GRADIENT_ACCUMULATION
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
def _probabilities(model, loader, device):
    model.eval(); result = []
    for batch in loader:
        logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        result.append(torch.sigmoid(logits).cpu().numpy())
    return np.vstack(result)


def _initialise_fold(fold, device):
    base_checkpoint = BASE_DIR / f"fold{fold}_model.pt"
    if not base_checkpoint.exists():
        raise FileNotFoundError(
            f"Missing {base_checkpoint}. Run `python main.py --mode factor-cv` first."
        )
    model = MentalRobertaMHLATModel(initialise_labels=False)
    incompatible = model.load_state_dict(
        torch.load(base_checkpoint, map_location="cpu"), strict=False
    )
    unexpected = list(incompatible.unexpected_keys)
    if unexpected:
        raise RuntimeError(f"Unexpected keys while transferring factor fold {fold}: {unexpected}")
    allowed = {"hop_projection.weight", "hop_projection.bias", "hop_gate",
               "hop_norm.weight", "hop_norm.bias"}
    if not set(incompatible.missing_keys).issubset(allowed):
        raise RuntimeError(f"Unexpected missing keys: {incompatible.missing_keys}")
    return model.to(device)


def _train_fold(fold, train_idx, valid_idx, dataset, targets, device):
    checkpoint, prediction_file = _fold_paths(fold)
    if checkpoint.exists() and prediction_file.exists():
        saved = np.load(prediction_file, allow_pickle=True)
        summary = json.loads(str(saved["summary"]))
        if (np.array_equal(saved["valid_indices"], valid_idx)
                and summary.get("training_version") == TRAINING_VERSION):
            print(f"MHLAT-v4 fold {fold}: resumed")
            return saved["probabilities"].astype(np.float32), summary

    seed_everything(config.SEED + 500 + fold)
    train_loader = _loader(dataset, train_idx, True)
    valid_loader = _loader(dataset, valid_idx, False)
    model = _initialise_fold(fold, device)
    loss_fn, centre_weights = _loss(dataset, train_idx, device)
    optimizer = AdamW(mhlat_optimizer_parameters(model), weight_decay=config.WEIGHT_DECAY)
    updates = int(np.ceil(len(train_loader) / config.GRADIENT_ACCUMULATION))
    updates *= config.FACTOR_MHLAT_EPOCHS
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, max(1, int(updates * config.WARMUP_RATIO)), max(1, updates)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=config.FP16)
    prevalence = targets[train_idx].mean(0)

    initial = _probabilities(model, valid_loader, device)
    initial_score = f1_score(
        targets[valid_idx], apply_prior_topk(initial, prevalence, 1.10),
        average="macro", zero_division=0,
    )
    best_probability = initial.copy()
    best = {
        "fold": fold, "epoch": 0, "score": float(initial_score),
        "training_version": TRAINING_VERSION,
    }
    torch.save(model.state_dict(), checkpoint)
    print(f"MHLAT-v4 fold={fold} epoch=0 transferred_macro_f1={initial_score:.4f}")

    for epoch in range(1, config.FACTOR_MHLAT_EPOCHS + 1):
        loss = _train_epoch(
            model, train_loader, loss_fn, centre_weights, optimizer, scheduler,
            scaler, device, epoch,
        )
        probability = _probabilities(model, valid_loader, device)
        score = f1_score(
            targets[valid_idx], apply_prior_topk(probability, prevalence, 1.10),
            average="macro", zero_division=0,
        )
        print(f"MHLAT-v4 fold={fold} epoch={epoch} loss={loss:.4f} macro_f1={score:.4f}")
        if score > best["score"]:
            best_probability = probability.copy()
            best = {
                "fold": fold, "epoch": epoch, "score": float(score),
                "train_loss": loss, "training_version": TRAINING_VERSION,
            }
            torch.save(model.state_dict(), checkpoint)

    np.savez_compressed(
        prediction_file, probabilities=best_probability, valid_indices=valid_idx,
        summary=json.dumps(best),
    )
    del model, optimizer, scheduler, train_loader, valid_loader
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return best_probability, best


def _current_components():
    required = [BASE_OOF_FILE, OLD_CROSS_OOF_FILE, V3_OOF_FILE, V3_CALIBRATION_FILE]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing accepted Task 2 artefacts: {missing}")
    base_saved = np.load(BASE_OOF_FILE)
    base = (
        config.FACTOR_SEMANTIC_MODEL_WEIGHT * base_saved["semantic"]
        + config.FACTOR_CPU_ENSEMBLE_WEIGHT * base_saved["cpu"]
    )
    old_cross = np.load(OLD_CROSS_OOF_FILE)["probabilities"]
    v3 = np.load(V3_OOF_FILE)["probabilities"]
    calibration = json.loads(V3_CALIBRATION_FILE.read_text(encoding="utf-8"))
    current = (
        float(calibration["base_weight"]) * base
        + float(calibration["old_cross_weight"]) * old_cross
        + float(calibration["new_cross_weight"]) * v3
    )
    return current.astype(np.float32), base_saved["targets"].astype(np.int8), calibration


def _blend_grid(current, mhlat, targets, indices, prevalence):
    rows = []
    for weight in (0.0, 0.10, 0.20, 0.30, 0.40, 0.50):
        probability = (1.0 - weight) * current[indices] + weight * mhlat[indices]
        for ratio in (1.0, 1.10, 1.25):
            score = f1_score(
                targets[indices], apply_prior_topk(probability, prevalence, ratio),
                average="macro", zero_division=0,
            )
            rows.append({"mhlat_weight": weight, "prevalence_ratio": ratio,
                         "macro_f1": float(score)})
    return sorted(rows, key=lambda item: item["macro_f1"], reverse=True)


def train_factor_mhlat_v4(only_fold0=False):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cache = build_factor_cache(train=True)
    dataset = FactorDataset(cache)
    frame = load_train_data().reset_index(drop=True)
    current, targets, current_calibration = _current_components()
    folds = list(StratifiedGroupKFold(
        n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED
    ).split(np.zeros(len(frame)), frame.risk_label, frame.anon_user_id))
    device = torch.device(config.DEVICE)
    mhlat_oof = np.zeros_like(current, dtype=np.float32)
    summaries = []
    selected = folds[:1] if only_fold0 else folds
    for fold, (train_idx, valid_idx) in enumerate(selected):
        probability, summary = _train_fold(
            fold, train_idx, valid_idx, dataset, targets, device
        )
        mhlat_oof[valid_idx] = probability
        summaries.append(summary)

    if only_fold0:
        fit, valid = folds[0]
        grid = _blend_grid(current, mhlat_oof, targets, valid, targets[fit].mean(0))
        baseline = f1_score(
            targets[valid], apply_prior_topk(
                current[valid], targets[fit].mean(0),
                float(current_calibration["prevalence_ratio"]),
            ), average="macro", zero_division=0,
        )
        result = {
            "training_version": TRAINING_VERSION, "folds": summaries,
            "current_fixed_fold0": float(baseline),
            "best_fold0_optimistic": grid[0], "top10": grid[:10],
        }
        (OUTPUT_DIR / "fold0_ablation.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        print(json.dumps(result, indent=2)); return result

    full_grid = _blend_grid(
        current, mhlat_oof, targets, np.arange(len(targets)), targets.mean(0)
    )
    nested_prediction = np.zeros_like(targets, dtype=bool)
    baseline_prediction = np.zeros_like(targets, dtype=bool)
    parameters = []
    for fold, (fit, valid) in enumerate(folds):
        parameter = _blend_grid(
            current, mhlat_oof, targets, fit, targets[fit].mean(0)
        )[0]
        mixed = (
            (1.0 - parameter["mhlat_weight"]) * current[valid]
            + parameter["mhlat_weight"] * mhlat_oof[valid]
        )
        nested_prediction[valid] = apply_prior_topk(
            mixed, targets[fit].mean(0), parameter["prevalence_ratio"]
        )
        baseline_prediction[valid] = apply_prior_topk(
            current[valid], targets[fit].mean(0),
            float(current_calibration["prevalence_ratio"]),
        )
        parameters.append({"fold": fold, **parameter})
    nested_score = f1_score(
        targets, nested_prediction, average="macro", zero_division=0
    )
    baseline_nested = f1_score(
        targets, baseline_prediction, average="macro", zero_division=0
    )
    production_weight = float(np.median([x["mhlat_weight"] for x in parameters]))
    production_ratio = float(np.median([x["prevalence_ratio"] for x in parameters]))
    production_probability = (
        (1.0 - production_weight) * current + production_weight * mhlat_oof
    )
    production_score = f1_score(
        targets, apply_prior_topk(production_probability, targets.mean(0), production_ratio),
        average="macro", zero_division=0,
    )
    # Compare the residual expert against the exact same decoder ratio. This
    # prevents a ratio change from being misattributed to MHLAT itself.
    baseline_same_ratio = f1_score(
        targets, apply_prior_topk(current, targets.mean(0), production_ratio),
        average="macro", zero_division=0,
    )
    baseline_full = f1_score(
        targets, apply_prior_topk(
            current, targets.mean(0), float(current_calibration["prevalence_ratio"])
        ), average="macro", zero_division=0,
    )
    adopted = bool(
        production_weight > 0
        and nested_score >= baseline_nested + 0.003
        and production_score >= baseline_full
        and production_score >= baseline_same_ratio + 0.001
    )
    calibration = {
        "training_version": TRAINING_VERSION, "adopted": adopted,
        "mhlat_weight": production_weight,
        "existing_weight": 1.0 - production_weight,
        "prevalence_ratio": production_ratio,
        "nested_macro_f1": float(nested_score),
        "baseline_nested_macro_f1": float(baseline_nested),
        "production_oof_macro_f1": float(production_score),
        "baseline_production_oof_macro_f1": float(baseline_full),
        "baseline_same_ratio_oof_macro_f1": float(baseline_same_ratio),
        "training_prevalence": targets.mean(0).tolist(),
    }
    result = {
        "training_version": TRAINING_VERSION, "folds": summaries,
        "crossfit_parameters": parameters, "calibration": calibration,
        "best_full_oof_optimistic": full_grid[0], "top10": full_grid[:10],
    }
    RESULT_FILE.write_text(json.dumps(result, indent=2), encoding="utf-8")
    CALIBRATION_FILE.write_text(json.dumps(calibration, indent=2), encoding="utf-8")
    np.savez_compressed(OOF_FILE, probabilities=mhlat_oof, targets=targets)
    print(json.dumps(result, indent=2)); return result


if __name__ == "__main__":
    train_factor_mhlat_v4()
