"""Leaderboard inference for the fold-0 V38 Task 2 experiment.

V38 is deliberately a leaderboard probe rather than a production model: the
only trained checkpoint is fold 0.  It replaces 25% of the five-fold semantic
MentalRoBERTa component, matching the strict ablation that improved fold 0.
"""
from __future__ import annotations

import json

import numpy as np
import torch
from torch.utils.data import DataLoader

from configs.config import config
from datasets.factor_cache_builder import build_factor_cache
from models.factor_definition_retrieval_v38 import DefinitionRetrievalFactorModel
from models.factor_model import MentalRobertaFactorModel
from trainer.factor_definition_retrieval_v38 import (
    BASE_CHECKPOINT, CHECKPOINT, FEATURES, TOPK, _collate_indexed, IndexedDataset,
)
from trainer.factor_train import FactorDataset, _collate


OUTPUT = config.OUTPUT_DIR / "factor_definition_retrieval_v38"
TEST_CACHE = OUTPUT / "leaderboard_probabilities.npz"
TEST_FEATURES = OUTPUT / "leaderboard_base_features.npz"
REPLACEMENT_WEIGHT = 0.25


def _signature():
    return json.dumps({
        "version": "factor-integrated-definition-retrieval-v38-leaderboard",
        "checkpoint": [int(CHECKPOINT.stat().st_size), int(CHECKPOINT.stat().st_mtime_ns)],
        "base_checkpoint": [int(BASE_CHECKPOINT.stat().st_size), int(BASE_CHECKPOINT.stat().st_mtime_ns)],
        "topk": TOPK,
    }, sort_keys=True)


@torch.no_grad()
def _encode_features(dataset, device):
    if TEST_FEATURES.exists():
        saved = np.load(TEST_FEATURES)
        if saved["features"].shape[0] == len(dataset):
            return saved["features"].astype(np.float32)
    model = MentalRobertaFactorModel(initialise_labels=False).to(device)
    model.load_state_dict(torch.load(BASE_CHECKPOINT, map_location=device)); model.eval()
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=_collate)
    values = []
    for position, batch in enumerate(loader, 1):
        _, feature = model(
            batch["input_ids"].to(device), batch["attention_mask"].to(device),
            return_features=True,
        )
        values.append(feature.cpu().numpy())
        if position % 50 == 0 or position == len(loader):
            print(f"V38 leaderboard base features: {position}/{len(loader)}", flush=True)
    result = np.vstack(values).astype(np.float32)
    np.savez_compressed(TEST_FEATURES, features=result)
    del model
    if device.type == "cuda": torch.cuda.empty_cache()
    return result


def _retrieval(train_features, test_features):
    train = train_features / np.maximum(np.linalg.norm(train_features, axis=1, keepdims=True), 1e-8)
    test = test_features / np.maximum(np.linalg.norm(test_features, axis=1, keepdims=True), 1e-8)
    similarity = test @ train.T
    count = min(TOPK, len(train_features))
    chosen = np.argpartition(similarity, -count, axis=1)[:, -count:]
    ordered = np.take_along_axis(
        chosen, np.argsort(np.take_along_axis(similarity, chosen, axis=1), axis=1)[:, ::-1], axis=1,
    )
    return torch.tensor(train_features[ordered], dtype=torch.float32)


@torch.no_grad()
def v38_factor_probabilities(row_ids, force=False):
    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Missing V38 checkpoint: {CHECKPOINT}. Run --mode factor-definition-retrieval-v38-fold0 first."
        )
    row_ids = np.asarray(row_ids).astype(str)
    signature = _signature()
    if TEST_CACHE.exists() and not force:
        saved = np.load(TEST_CACHE)
        if (saved["probabilities"].shape == (len(row_ids), config.NUM_FACTORS)
                and np.array_equal(saved["row_ids"].astype(str), row_ids)
                and str(saved["signature"]) == signature):
            print("Loaded cached V38 leaderboard probabilities", flush=True)
            return saved["probabilities"].astype(np.float32), REPLACEMENT_WEIGHT

    device = torch.device(config.DEVICE)
    train_dataset = FactorDataset(build_factor_cache(train=True))
    test_dataset = FactorDataset(build_factor_cache(train=False))
    if len(test_dataset) != len(row_ids):
        raise ValueError("V38 test cache and submission row IDs have different lengths")
    train_features = np.load(FEATURES)["features"].astype(np.float32)
    if len(train_features) != len(train_dataset):
        raise ValueError("V38 train feature cache is stale")
    test_features = _encode_features(test_dataset, device)
    retrieval = _retrieval(train_features, test_features)

    state = torch.load(CHECKPOINT, map_location="cpu")
    model = DefinitionRetrievalFactorModel(
        state["definition_embeddings"], state["tail_mask"].bool(), state["feature_std"],
    ).to(device)
    model.load_state_dict(state); model.eval()
    indexed = IndexedDataset(test_dataset, np.arange(len(test_dataset)), retrieval)
    loader = DataLoader(
        indexed, batch_size=config.BATCH_SIZE, shuffle=False,
        collate_fn=_collate_indexed, num_workers=config.NUM_WORKERS,
    )
    result = []
    for position, batch in enumerate(loader, 1):
        logits = model(
            batch["input_ids"].to(device), batch["attention_mask"].to(device),
            batch["retrieval"].to(device),
        )
        result.append(torch.sigmoid(logits).cpu().numpy())
        if position % 50 == 0 or position == len(loader):
            print(f"V38 leaderboard prediction: {position}/{len(loader)}", flush=True)
    probabilities = np.vstack(result).astype(np.float32)
    np.savez_compressed(
        TEST_CACHE, probabilities=probabilities, row_ids=row_ids, signature=signature,
    )
    del model
    if device.type == "cuda": torch.cuda.empty_cache()
    return probabilities, REPLACEMENT_WEIGHT


__all__ = ["v38_factor_probabilities", "REPLACEMENT_WEIGHT"]
