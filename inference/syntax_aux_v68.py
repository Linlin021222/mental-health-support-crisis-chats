"""Inference for the strictly adopted V68 Task-2 syntax expert."""
from __future__ import annotations
import json
import joblib
import numpy as np
from configs.config import config
from preprocess.style_syntax import transform_lower_syntax
from trainer.case_syntax_v66 import _factor_probability
from trainer.syntax_aux_v68 import OUTPUT,RESULTS


def syntax_factor_probabilities(texts):
    if not RESULTS.exists(): return None
    payload=json.loads(RESULTS.read_text(encoding="utf-8"))
    # V68 improved strict local OOF but reduced the official leaderboard
    # Task-2 score (0.6073 -> 0.6015).  Keep the artefacts for analysis, but do
    # not silently deploy it again unless a later experiment explicitly marks
    # it as production-safe.
    if not payload.get("production_adopted",False): return None
    rows=[]
    for fold in range(config.N_FOLDS):
        path=OUTPUT/f"fold{fold}.joblib"
        if not path.exists(): return None
        artifact=joblib.load(path)
        rows.append(_factor_probability(artifact["models"],transform_lower_syntax(artifact["vectorizer"],texts)))
    return np.mean(rows,axis=0).astype(np.float32)
