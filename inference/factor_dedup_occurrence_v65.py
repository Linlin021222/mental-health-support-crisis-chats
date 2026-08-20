"""Inference for the strictly gated V65 sparse fold ensemble."""
from __future__ import annotations

import json

import joblib
import numpy as np

from baseline import _factor_probabilities
from configs.config import config


OUTPUT = config.OUTPUT_DIR / "factor_dedup_occurrence_v65"
CALIBRATION = OUTPUT / "calibration.json"


def dedup_occurrence_probabilities(texts):
    if not CALIBRATION.exists():
        return None
    calibration = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    if not calibration.get("adopted", False):
        return None
    probabilities = []
    for fold in range(config.N_FOLDS):
        path = OUTPUT / f"fold{fold}.joblib"
        if not path.exists():
            return None
        artifact = joblib.load(path)
        matrix = artifact["vectorizer"].transform(list(map(str, texts)))
        probabilities.append(_factor_probabilities(artifact["models"], matrix))
    return np.mean(probabilities, axis=0).astype(np.float32)
