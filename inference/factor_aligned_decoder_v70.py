"""Inference for the strictly label-local V70 decoder experiment."""
from __future__ import annotations

import json
import numpy as np

from configs.config import config
from trainer.factor_aligned_decoder_v70 import CALIBRATION, DEFINITION_GUARDS


def _rank(values):
    order = np.argsort(values, kind="stable")
    result = np.empty(len(values), dtype=np.float32)
    result[order] = np.arange(len(values), dtype=np.float32) / max(1, len(values)-1)
    return result


def _topk(values, prevalence, ratio):
    n = len(values)
    count = max(1, min(n, int(round(n*float(prevalence)*float(ratio)))))
    selected = np.argpartition(values, n-count)[n-count:]
    result = np.zeros(n, dtype=bool); result[selected] = True
    return result


def apply_v70(probabilities, predictions, texts=None, force=False):
    if not CALIBRATION.exists(): return predictions, False
    data = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    allowed = (data.get("experimental_adopted", False) if force
               else data.get("production_adopted", False))
    if not allowed: return predictions, False
    original = np.asarray(predictions, dtype=bool)
    result = original.copy()
    probability = np.asarray(probabilities, dtype=np.float32)
    prevalence = np.asarray(data["training_prevalence"], dtype=np.float32)
    for name, ratio in data["repairs"].items():
        label = config.FACTOR2ID[name]
        result[:, label] = _topk(
            _rank(probability[:, label]), prevalence[label], ratio,
        )
        if texts is not None and name in DEFINITION_GUARDS:
            protected = np.asarray([
                bool(DEFINITION_GUARDS[name].search(str(text))) for text in texts
            ], dtype=bool)
            result[:, label] |= original[:, label] & protected
    return result, True
