"""Conservative leaderboard gate for the full-data Qwen3-8B risk expert.

The Qwen model was trained on all 1,635 labelled posts on Kaggle.  Its four
class probabilities are used only when the temperature-scaled prediction is
high-confidence.  Evidence continues to come from the independently trained
label-conditional evidence ensemble in :mod:`inference.predict`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from configs.config import config
from preprocess.preprocess import load_test_data


PROBABILITIES = (
    config.OUTPUT_DIR / "task1_qwen3_8b_full" / "test_probabilities.npz"
)
TEMPERATURE = 0.70
CONFIDENCE_THRESHOLD = 0.70
BLEND_WEIGHT = 0.30
ATTEMPT_CALIBRATED_TEMPERATURE = 1.20
ATTEMPT_CALIBRATED_BLEND_WEIGHT = 0.40
ATTEMPT_CALIBRATED_CLASS_BIAS = (0.0, 0.0, -0.15, 0.30)


def _temperature_scale(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    logits = np.log(np.clip(np.asarray(probabilities, dtype=np.float64), 1e-8, 1.0))
    logits /= float(temperature)
    logits -= logits.max(axis=1, keepdims=True)
    scaled = np.exp(logits)
    return (scaled / scaled.sum(axis=1, keepdims=True)).astype(np.float32)


def qwen3_8b_probabilities(
    path: Path = PROBABILITIES,
    temperature: float = TEMPERATURE,
):
    """Return validated row IDs and temperature-scaled Qwen probabilities."""
    if not path.exists():
        raise FileNotFoundError(
            f"Missing Qwen3-8B probabilities: {path}. Copy the Kaggle full-run "
            "NPZ to this location before using predict-qwen3-8b."
        )
    with np.load(path, allow_pickle=True) as saved:
        row_ids = np.asarray(saved["row_id"]).astype(str)
        probabilities = np.asarray(saved["probabilities"], dtype=np.float32)
    expected = load_test_data().row_id.astype(str).to_numpy()
    if probabilities.shape != (len(expected), config.NUM_RISK_CLASSES):
        raise ValueError(
            f"Qwen probability shape {probabilities.shape} != "
            f"({len(expected)}, {config.NUM_RISK_CLASSES})"
        )
    if not np.array_equal(row_ids, expected):
        raise ValueError("Qwen row IDs/order differ from leaderboard.xlsx")
    if not np.isfinite(probabilities).all():
        raise ValueError("Qwen probabilities contain NaN or infinity")
    return row_ids, _temperature_scale(probabilities, temperature)


__all__ = [
    "ATTEMPT_CALIBRATED_BLEND_WEIGHT", "ATTEMPT_CALIBRATED_CLASS_BIAS",
    "ATTEMPT_CALIBRATED_TEMPERATURE", "BLEND_WEIGHT", "CONFIDENCE_THRESHOLD", "TEMPERATURE",
    "qwen3_8b_probabilities",
]
