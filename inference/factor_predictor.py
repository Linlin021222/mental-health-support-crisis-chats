"""Inference helper for the standalone MentalRoBERTa factor model."""
import numpy as np
import torch
from torch.utils.data import DataLoader

from configs.config import config
from datasets.factor_cache_builder import build_factor_cache
from models.factor_model import MentalRobertaFactorModel
from trainer.factor_train import FactorDataset, _collate, FULL_CHECKPOINT

LEGACY_FULL_CHECKPOINT = config.OUTPUT_DIR / "factor_full_model_asl.pt"
CV_CHECKPOINTS = [config.OUTPUT_DIR / "factor_cv" / f"fold{i}_model.pt"
                  for i in range(config.N_FOLDS)]
_STANDALONE_CACHE = None


@torch.no_grad()
def _checkpoint_probabilities(checkpoint, loader, device):
    model = MentalRobertaFactorModel(initialise_labels=False).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    probabilities = []
    for batch in loader:
        logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        probabilities.append(torch.sigmoid(logits).cpu().numpy())
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return np.vstack(probabilities)


@torch.no_grad()
def standalone_factor_probabilities():
    """Return the validated semantic/legacy neural ensemble.

    The returned probabilities are normalized over the neural portion. The
    caller subsequently gives this mixture 75% and the CPU lexical model 25%,
    yielding the strict-selected 0.50/0.25/0.25 final weights.
    """
    global _STANDALONE_CACHE
    if _STANDALONE_CACHE is not None:
        return _STANDALONE_CACHE
    if not FULL_CHECKPOINT.exists():
        return None, None
    cache = build_factor_cache(train=False)
    dataset = FactorDataset(cache)
    loader = DataLoader(
        dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=_collate,
        num_workers=config.NUM_WORKERS,
    )
    device = torch.device(config.DEVICE)
    if config.FACTOR_USE_CV_ENSEMBLE and all(path.exists() for path in CV_CHECKPOINTS):
        probabilities = np.mean([
            _checkpoint_probabilities(path, loader, device) for path in CV_CHECKPOINTS
        ], axis=0)
        print(f"Using {len(CV_CHECKPOINTS)}-fold user-disjoint MentalRoBERTa ensemble")
        row_ids = [x["row_id"] for x in dataset.data]
        _STANDALONE_CACHE = (row_ids, probabilities.astype(np.float32))
        return _STANDALONE_CACHE

    semantic = _checkpoint_probabilities(FULL_CHECKPOINT, loader, device)
    if LEGACY_FULL_CHECKPOINT.exists():
        legacy = _checkpoint_probabilities(LEGACY_FULL_CHECKPOINT, loader, device)
        semantic_weight = float(config.FACTOR_SEMANTIC_MODEL_WEIGHT)
        legacy_weight = float(config.FACTOR_LEGACY_MODEL_WEIGHT)
        probabilities = (
            semantic_weight * semantic + legacy_weight * legacy
        ) / (semantic_weight + legacy_weight)
    else:
        probabilities = semantic
        print(f"Warning: legacy Task 2 checkpoint missing: {LEGACY_FULL_CHECKPOINT}")
    row_ids = [x["row_id"] for x in dataset.data]
    _STANDALONE_CACHE = (row_ids, probabilities)
    return _STANDALONE_CACHE
