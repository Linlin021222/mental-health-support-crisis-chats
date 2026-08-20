"""Five-fold user-disjoint shared text/label cross-encoder for Task 2."""
import json
import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification, AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

from configs.config import config
from inference.factor_nli import TRAIN_NLI_FILE, _entailment_index, _rank_decode
from preprocess.preprocess import load_train_data
from trainer.factor_cv import OOF_FILE
from trainer.factor_cross_encoder_ablation import (
    OUTPUT_DIR, PairDataset, _training_pairs, _collator, _predict,
)
from utils.seed import seed_everything


CROSS_OOF_FILE = OUTPUT_DIR / "oof_predictions.npz"
CROSS_RESULTS_FILE = OUTPUT_DIR / "cv_results.json"
CROSS_CALIBRATION_FILE = OUTPUT_DIR / "calibration.json"


def _fold_paths(fold):
    return OUTPUT_DIR / f"fold{fold}_model.pt", OUTPUT_DIR / f"fold{fold}_valid.npz"


def _train_fold(fold, train_idx, valid_idx, frame, targets, tokenizer, device):
    checkpoint, prediction_file = _fold_paths(fold)
    if checkpoint.exists() and prediction_file.exists():
        saved = np.load(prediction_file)
        if np.array_equal(saved["valid_indices"], valid_idx):
            print(f"cross-encoder fold {fold}: resumed")
            if "summary" in saved.files:
                summary = json.loads(str(saved["summary"]))
            else:
                ablation = json.loads(
                    (OUTPUT_DIR / "fold0_ablation.json").read_text(encoding="utf-8")
                )
                summary = {
                    "fold": fold, "epoch": ablation["best_epoch"],
                    "score": ablation["comparisons"]["cross_encoder"],
                    "pair_count": ablation["pair_count"], "resumed_ablation": True,
                }
            return saved["probabilities"].astype(np.float32), summary

    seed_everything(config.SEED + fold)
    train_targets = targets[train_idx]
    pairs = _training_pairs(train_targets, config.SEED + fold)
    train_texts = frame.text.iloc[train_idx].tolist()
    loader = DataLoader(
        PairDataset(train_texts, pairs),
        batch_size=config.FACTOR_CROSS_ENCODER_BATCH_SIZE, shuffle=True,
        collate_fn=_collator(tokenizer), num_workers=0,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        config.FACTOR_NLI_MODEL_NAME, dtype=torch.float32,
    ).to(device)
    model.gradient_checkpointing_enable(); model.config.use_cache = False
    entailment = _entailment_index(model)
    optimizer = AdamW(model.parameters(), lr=config.FACTOR_CROSS_ENCODER_LR,
                      weight_decay=config.WEIGHT_DECAY)
    update_steps = int(np.ceil(len(loader) / config.FACTOR_CROSS_ENCODER_ACCUMULATION))
    total_steps = update_steps * config.FACTOR_CROSS_ENCODER_EPOCHS
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, max(1, int(total_steps * config.WARMUP_RATIO)), total_steps
    )
    scaler = torch.amp.GradScaler("cuda", enabled=config.FP16)
    prevalence = train_targets.mean(0)
    valid_texts = frame.text.iloc[valid_idx].tolist()
    valid_targets = targets[valid_idx]
    best = {"fold": fold, "epoch": 0, "score": -1.0, "pair_count": len(pairs)}
    best_probability = None
    print(f"cross-encoder fold {fold}: train={len(train_idx)} pairs={len(pairs)} "
          f"valid={len(valid_idx)} overlap=0")
    for epoch in range(1, config.FACTOR_CROSS_ENCODER_EPOCHS + 1):
        model.train(); optimizer.zero_grad(set_to_none=True); losses = []
        for step, batch in enumerate(tqdm(
            loader, desc=f"cross fold {fold} epoch {epoch}"
        ), 1):
            binary = batch.pop("targets").to(device)
            weights = batch.pop("weights").to(device)
            batch = {key: value.to(device) for key, value in batch.items()}
            class_targets = torch.where(
                binary > 0, torch.full_like(binary, entailment),
                torch.full_like(binary, 1 - entailment),
            )
            with torch.autocast(device_type=device.type, enabled=config.FP16):
                raw = torch.nn.functional.cross_entropy(
                    model(**batch).logits, class_targets, reduction="none"
                )
                loss = (raw * weights).mean() / config.FACTOR_CROSS_ENCODER_ACCUMULATION
            scaler.scale(loss).backward()
            losses.append(float(loss.detach()) * config.FACTOR_CROSS_ENCODER_ACCUMULATION)
            if (step % config.FACTOR_CROSS_ENCODER_ACCUMULATION == 0
                    or step == len(loader)):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
                old_scale = scaler.get_scale()
                scaler.step(optimizer); scaler.update()
                if scaler.get_scale() >= old_scale:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
        probability = _predict(model, tokenizer, valid_texts, device)
        score = f1_score(
            valid_targets, _rank_decode(probability, prevalence, 1.0),
            average="macro", zero_division=0,
        )
        print(f"cross fold={fold} epoch={epoch} loss={np.mean(losses):.4f} "
              f"macro_f1={score:.4f}")
        if score > best["score"]:
            best.update(epoch=epoch, score=float(score), loss=float(np.mean(losses)))
            best_probability = probability.copy()
            torch.save(model.state_dict(), checkpoint)
    np.savez_compressed(
        prediction_file, probabilities=best_probability, valid_indices=valid_idx,
        summary=json.dumps(best),
    )
    del model, optimizer, scheduler, loader
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return best_probability, best


def _component_grid(base, nli, cross, targets, indices, prevalence):
    candidates = []
    for cross_weight in np.linspace(0.0, 0.8, 9):
        for nli_weight in np.linspace(0.0, 0.4, 5):
            if cross_weight + nli_weight > 1.0 + 1e-9:
                continue
            base_weight = 1.0 - cross_weight - nli_weight
            probability = (base_weight * base[indices]
                           + nli_weight * nli[indices]
                           + cross_weight * cross[indices])
            for ratio in (0.80, 0.90, 1.00, 1.10, 1.25):
                prediction = _rank_decode(probability, prevalence, ratio)
                score = f1_score(
                    targets[indices], prediction, average="macro", zero_division=0
                )
                candidates.append({
                    "macro_f1": float(score), "base_weight": float(base_weight),
                    "nli_weight": float(nli_weight), "cross_weight": float(cross_weight),
                    "prevalence_ratio": float(ratio),
                })
    candidates.sort(key=lambda x: x["macro_f1"], reverse=True)
    return candidates


def train_cross_encoder_cv():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = load_train_data().reset_index(drop=True)
    targets = np.vstack(frame.factor_vector.to_numpy()).astype(np.int8)
    folds = list(StratifiedGroupKFold(
        n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED
    ).split(np.zeros(len(frame)), frame.risk_label, frame.anon_user_id))
    tokenizer = AutoTokenizer.from_pretrained(config.FACTOR_NLI_MODEL_NAME, use_fast=True)
    device = torch.device(config.DEVICE)
    cross_oof = np.zeros_like(targets, dtype=np.float32)
    summaries = []
    for fold, (train_idx, valid_idx) in enumerate(folds):
        probability, summary = _train_fold(
            fold, train_idx, valid_idx, frame, targets, tokenizer, device
        )
        cross_oof[valid_idx] = probability; summaries.append(summary)

    saved = np.load(OOF_FILE)
    base = (config.FACTOR_SEMANTIC_MODEL_WEIGHT * saved["semantic"]
            + config.FACTOR_CPU_ENSEMBLE_WEIGHT * saved["cpu"])
    nli = np.load(TRAIN_NLI_FILE)["probabilities"]
    prevalence = targets.mean(0)
    full_grid = _component_grid(
        base, nli, cross_oof, targets, np.arange(len(frame)), prevalence
    )

    crossfit_prediction = np.zeros_like(targets, dtype=bool)
    crossfit_parameters = []
    for fold, (fit, valid) in enumerate(folds):
        parameter = _component_grid(
            base, nli, cross_oof, targets, fit, targets[fit].mean(0)
        )[0]
        probability = (parameter["base_weight"] * base[valid]
                       + parameter["nli_weight"] * nli[valid]
                       + parameter["cross_weight"] * cross_oof[valid])
        crossfit_prediction[valid] = _rank_decode(
            probability, targets[fit].mean(0), parameter["prevalence_ratio"]
        )
        crossfit_parameters.append({"fold": fold, **parameter})
    crossfit_score = f1_score(
        targets, crossfit_prediction, average="macro", zero_division=0
    )
    cross_only_score = f1_score(
        targets, _rank_decode(cross_oof, prevalence, 1.0),
        average="macro", zero_division=0,
    )
    baseline_score = f1_score(
        targets, _rank_decode(base, prevalence, config.FACTOR_TOPK_RATIO),
        average="macro", zero_division=0,
    )
    best = full_grid[0]
    result = {
        "folds": summaries, "baseline_macro_f1": float(baseline_score),
        "cross_encoder_macro_f1": float(cross_only_score),
        "three_component_crossfit_macro_f1": float(crossfit_score),
        "crossfit_parameters": crossfit_parameters,
        "best_full_oof_optimistic": best, "top10_full_oof": full_grid[:10],
    }
    CROSS_RESULTS_FILE.write_text(json.dumps(result, indent=2), encoding="utf-8")
    calibration = {
        "model_name": config.FACTOR_NLI_MODEL_NAME,
        "base_weight": best["base_weight"], "nli_weight": best["nli_weight"],
        "cross_weight": best["cross_weight"],
        "prevalence_ratio": best["prevalence_ratio"],
        "training_prevalence": prevalence.tolist(),
        "crossfit_macro_f1": float(crossfit_score),
    }
    CROSS_CALIBRATION_FILE.write_text(json.dumps(calibration, indent=2), encoding="utf-8")
    np.savez_compressed(CROSS_OOF_FILE, probabilities=cross_oof, targets=targets)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    train_cross_encoder_cv()
