"""Inference for the strictly adopted Task-2 V66 case/syntax expert."""
from __future__ import annotations

import json

import joblib
import numpy as np

from configs.config import config
from preprocess.style_syntax import transform_cased
from trainer.case_syntax_v66 import _factor_probability, OUTPUT, RESULTS


def case_syntax_factor_probabilities(texts):
    # The controlled V66 ablation showed that casing alone hurts and the
    # slightly stronger lowercase+syntax V68 supersedes this branch.
    ablation = OUTPUT / "ablation.json"
    if ablation.exists():
        compared = json.loads(ablation.read_text(encoding="utf-8"))["systems"]
        if (compared["lowercase_text_plus_syntax"]["macro_f1"]
                >= compared["case_sensitive_plus_syntax"]["macro_f1"]):
            return None
    if not RESULTS.exists():
        return None
    payload = json.loads(RESULTS.read_text(encoding="utf-8"))
    if not payload.get("task2", {}).get("adopted", False):
        return None
    rows = []
    for fold in range(config.N_FOLDS):
        path = OUTPUT / f"factor_fold{fold}.joblib"
        if not path.exists():
            return None
        artifact = joblib.load(path)
        matrix = transform_cased(artifact["vectorizer"], texts)
        rows.append(_factor_probability(artifact["models"], matrix))
    return np.mean(rows, axis=0).astype(np.float32)
