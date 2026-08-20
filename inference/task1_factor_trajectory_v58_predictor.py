"""Production predictions from the experimental V58 factor trajectory."""
from __future__ import annotations

import json

import numpy as np
import torch

from configs.config import config
from inference.factor_predictor import standalone_factor_probabilities
from preprocess.preprocess import load_test_data
from trainer.task1_factor_trajectory_v58 import (
    FIXED_WEIGHT, FULL_CHECKPOINTS, FULL_MANIFEST, FactorTrajectory, _predict, _sequences,
)


@torch.no_grad()
def task1_v58_probabilities():
    if not FULL_MANIFEST.exists() or not all(path.exists() for path in FULL_CHECKPOINTS):
        return None, None, 0.0
    manifest = json.loads(FULL_MANIFEST.read_text(encoding="utf-8"))
    if not manifest.get("experimental_leaderboard_override", False):
        return None, None, 0.0
    frame = load_test_data().reset_index(drop=True)
    row_ids, factors = standalone_factor_probabilities()
    if factors is None:
        return None, None, 0.0
    expected = frame.row_id.astype(str).tolist()
    if list(map(str, row_ids)) != expected:
        by_id = {str(row_id): probability for row_id, probability in zip(row_ids, factors)}
        factors = np.vstack([by_id[row_id] for row_id in expected])
    labels = np.zeros(len(frame), dtype=np.int64)
    indices = np.arange(len(frame), dtype=np.int64)
    sequences = _sequences(frame, indices, factors, labels)
    device = torch.device(config.DEVICE)
    probabilities = []
    for checkpoint in FULL_CHECKPOINTS:
        model = FactorTrajectory().to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device)); model.eval()
        probabilities.append(_predict(model, sequences, len(frame), device))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return expected, np.mean(probabilities, axis=0), float(manifest.get("risk_weight", FIXED_WEIGHT))


__all__ = ["task1_v58_probabilities"]
