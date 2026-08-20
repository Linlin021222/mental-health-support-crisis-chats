"""Fold-0 ablation for paper-aligned classifier prototypes + count salience."""
import json
import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.optim import AdamW

from configs.config import config
from datasets.factor_cache_builder import build_factor_cache
from inference.factor_nli import TRAIN_NLI_FILE, _rank_decode
from models.factor_model import factor_optimizer_parameters
from trainer.factor_cv import OOF_FILE
from trainer.factor_train import (
    FactorDataset, _loader, _loss_and_model, _probabilities, _train_epoch,
)
from utils.seed import seed_everything


V2_DIR = config.OUTPUT_DIR / "factor_v2"
CHECKPOINT = V2_DIR / "fold0_model.pt"
PREDICTION_FILE = V2_DIR / "fold0_valid.npz"
RESULT_FILE = V2_DIR / "fold0_ablation.json"


def train_factor_v2_fold0():
    seed_everything(config.SEED)
    V2_DIR.mkdir(parents=True, exist_ok=True)
    cache = build_factor_cache(train=True)
    dataset = FactorDataset(cache)
    targets = np.vstack([x["factor_vector"].numpy() for x in dataset.data])
    risk = np.asarray([x["risk_label"] for x in dataset.data])
    groups = np.asarray([x["anon_user_id"] for x in dataset.data])
    train_idx, valid_idx = next(StratifiedGroupKFold(
        n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED
    ).split(np.zeros(len(dataset)), risk, groups))
    prevalence = targets[train_idx].mean(0)
    device = torch.device(config.DEVICE)
    model, loss_fn = _loss_and_model(dataset, train_idx, device)
    optimizer = AdamW(factor_optimizer_parameters(model), weight_decay=config.WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=config.FP16)
    train_loader = _loader(dataset, train_idx, True)
    valid_loader = _loader(dataset, valid_idx, False)
    best, best_epoch, best_probability = -1.0, 0, None
    epochs = []
    print(f"factor V2 fold0: train={len(train_idx)} valid={len(valid_idx)} users overlap=0")
    for epoch in range(1, config.FACTOR_EPOCHS + 1):
        loss = _train_epoch(model, train_loader, loss_fn, optimizer, scaler, device, epoch)
        probability, valid_targets = _probabilities(model, valid_loader, device)
        prediction = _rank_decode(probability, prevalence, 1.10)
        score = f1_score(valid_targets, prediction, average="macro", zero_division=0)
        epochs.append({"epoch": epoch, "loss": float(loss), "macro_f1": float(score)})
        print(f"factor V2 fold0 epoch={epoch} loss={loss:.4f} macro_f1={score:.4f}")
        if score > best:
            best, best_epoch, best_probability = float(score), epoch, probability.copy()
            torch.save(model.state_dict(), CHECKPOINT)

    old = np.load(OOF_FILE)
    current_base = (config.FACTOR_SEMANTIC_MODEL_WEIGHT * old["semantic"][valid_idx]
                    + config.FACTOR_CPU_ENSEMBLE_WEIGHT * old["cpu"][valid_idx])
    v2_base = (config.FACTOR_SEMANTIC_MODEL_WEIGHT * best_probability
               + config.FACTOR_CPU_ENSEMBLE_WEIGHT * old["cpu"][valid_idx])
    valid_targets = targets[valid_idx]
    comparisons = {
        "old_neural": f1_score(
            valid_targets, _rank_decode(old["semantic"][valid_idx], prevalence, 1.10),
            average="macro", zero_division=0,
        ),
        "old_neural_cpu": f1_score(
            valid_targets, _rank_decode(current_base, prevalence, 1.10),
            average="macro", zero_division=0,
        ),
        "v2_neural": best,
        "v2_neural_cpu": f1_score(
            valid_targets, _rank_decode(v2_base, prevalence, 1.10),
            average="macro", zero_division=0,
        ),
    }
    if TRAIN_NLI_FILE.exists():
        nli = np.load(TRAIN_NLI_FILE)["probabilities"][valid_idx]
        comparisons["old_neural_cpu_nli_fixed"] = f1_score(
            valid_targets, _rank_decode(0.70 * current_base + 0.30 * nli, prevalence, 1.0),
            average="macro", zero_division=0,
        )
        comparisons["v2_neural_cpu_nli_fixed"] = f1_score(
            valid_targets, _rank_decode(0.70 * v2_base + 0.30 * nli, prevalence, 1.0),
            average="macro", zero_division=0,
        )
    comparisons = {key: float(value) for key, value in comparisons.items()}
    payload = {
        "best_epoch": best_epoch, "epochs": epochs, "comparisons": comparisons,
        "paper_definition_init": bool(config.FACTOR_PAPER_DEFINITION_INIT),
        "semantic_classifier_init": bool(config.FACTOR_SEMANTIC_CLASSIFIER_INIT),
        "occurrence_alpha": float(config.FACTOR_OCCURRENCE_ALPHA),
    }
    np.savez_compressed(PREDICTION_FILE, probabilities=best_probability,
                        valid_indices=valid_idx)
    RESULT_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return payload


if __name__ == "__main__":
    train_factor_v2_fold0()
