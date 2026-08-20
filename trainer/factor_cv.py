"""Five-fold user-disjoint Task 2 training, OOF calibration and ensembling."""
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.optim import AdamW
from tqdm import tqdm

from baseline import _vectorizer, _fit_factor_models, _factor_probabilities
from configs.config import config
from datasets.factor_cache_builder import build_factor_cache
from models.factor_model import factor_optimizer_parameters
from preprocess.preprocess import load_train_data
from trainer.factor_train import (
    FactorDataset, _loader, _loss_and_model, _probabilities, _train_epoch,
)
from utils.factor_calibration import apply_prior_topk, calibrate_factor_thresholds
from utils.seed import seed_everything


CV_DIR = config.OUTPUT_DIR / "factor_cv"
OOF_FILE = CV_DIR / "factor_oof_predictions.npz"
RESULTS_FILE = CV_DIR / "factor_cv_results.json"


def _fold_paths(fold):
    return CV_DIR / f"fold{fold}_model.pt", CV_DIR / f"fold{fold}_valid.npz"


def train_factor_cv():
    CV_DIR.mkdir(parents=True, exist_ok=True)
    cache = build_factor_cache(train=True)
    dataset = FactorDataset(cache)
    frame = load_train_data()
    targets = np.vstack(frame.factor_vector.to_numpy())
    risk = np.asarray([x["risk_label"] for x in dataset.data])
    groups = np.asarray([x["anon_user_id"] for x in dataset.data])
    folds = list(StratifiedGroupKFold(
        n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED
    ).split(np.zeros(len(dataset)), risk, groups))
    device = torch.device(config.DEVICE)
    oof = np.zeros_like(targets, dtype=np.float32)
    fold_summary = []

    for fold, (train_idx, valid_idx) in enumerate(folds):
        checkpoint, prediction_file = _fold_paths(fold)
        if checkpoint.exists() and prediction_file.exists():
            saved = np.load(prediction_file)
            if np.array_equal(saved["valid_indices"], valid_idx):
                oof[valid_idx] = saved["probabilities"]
                fold_summary.append(json.loads(str(saved["summary"])))
                print(f"factor CV fold {fold}: resumed")
                continue
        seed_everything(config.SEED + fold)
        model, loss_fn = _loss_and_model(dataset, train_idx, device)
        optimizer = AdamW(factor_optimizer_parameters(model), weight_decay=config.WEIGHT_DECAY)
        scaler = torch.amp.GradScaler("cuda", enabled=config.FP16)
        train_loader = _loader(dataset, train_idx, True)
        valid_loader = _loader(dataset, valid_idx, False)
        prevalence = targets[train_idx].mean(0)
        best = {"score": -1.0}
        best_probability = None
        print(f"factor CV fold {fold}: train={len(train_idx)} valid={len(valid_idx)} overlap=0")
        for epoch in range(1, config.FACTOR_EPOCHS + 1):
            loss = _train_epoch(model, train_loader, loss_fn, optimizer, scaler, device, epoch)
            probability, valid_targets = _probabilities(model, valid_loader, device)
            prediction = apply_prior_topk(probability, prevalence, ratio=1.10)
            score = f1_score(valid_targets, prediction, average="macro", zero_division=0)
            print(f"factor CV fold={fold} epoch={epoch} loss={loss:.4f} macro_f1={score:.4f}")
            if score > best["score"]:
                best = {"fold": fold, "epoch": epoch, "score": float(score)}
                best_probability = probability.copy()
                torch.save(model.state_dict(), checkpoint)
        oof[valid_idx] = best_probability
        np.savez_compressed(
            prediction_file, probabilities=best_probability, valid_indices=valid_idx,
            summary=json.dumps(best),
        )
        fold_summary.append(best)
        del model, loss_fn, optimizer
        if device.type == "cuda": torch.cuda.empty_cache()

    # Cross-fitted sparse predictions: every post is scored by a vectorizer
    # and 24 classifiers that did not see its user.
    cpu_oof = np.zeros_like(targets, dtype=np.float32)
    for fold, (train_idx, valid_idx) in enumerate(tqdm(folds, desc="CPU OOF")):
        vectorizer = _vectorizer()
        train_x = vectorizer.fit_transform(frame.text.iloc[train_idx])
        valid_x = vectorizer.transform(frame.text.iloc[valid_idx])
        cpu_oof[valid_idx] = _factor_probabilities(
            _fit_factor_models(train_x, targets[train_idx]), valid_x
        )
        del vectorizer, train_x, valid_x

    prevalence = targets.mean(0)
    grid = []
    for semantic_weight in np.linspace(0.0, 1.0, 11):
        probability = semantic_weight * oof + (1.0 - semantic_weight) * cpu_oof
        for ratio in (0.8, 0.9, 1.0, 1.1, 1.25, 1.4):
            prediction = apply_prior_topk(probability, prevalence, ratio)
            grid.append({
                "semantic_weight": float(semantic_weight), "topk_ratio": ratio,
                "macro_f1": float(f1_score(
                    targets, prediction, average="macro", zero_division=0
                )),
            })
    grid.sort(key=lambda x: x["macro_f1"], reverse=True)
    best_probability = (
        grid[0]["semantic_weight"] * oof
        + (1.0 - grid[0]["semantic_weight"]) * cpu_oof
    )
    thresholds = calibrate_factor_thresholds(targets, best_probability)
    threshold_score = f1_score(
        targets, best_probability >= thresholds[None, :], average="macro", zero_division=0
    )
    payload = {
        "folds": fold_summary,
        "mean_fold_macro_f1": float(np.mean([x["score"] for x in fold_summary])),
        "topk_best": grid[0], "topk_top15": grid[:15],
        "oof_fitted_threshold_macro_f1_optimistic": float(threshold_score),
        "oof_thresholds": thresholds.tolist(),
    }
    RESULTS_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    np.savez_compressed(
        OOF_FILE, semantic=oof, cpu=cpu_oof, targets=targets, thresholds=thresholds,
    )
    print(json.dumps(payload, indent=2))
    return payload


if __name__ == "__main__":
    train_factor_cv()
