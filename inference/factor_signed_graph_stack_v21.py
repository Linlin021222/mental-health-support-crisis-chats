"""Production inference for the adopted signed label-dependency V21 stack."""
from __future__ import annotations

import joblib
import numpy as np
from sklearn.preprocessing import normalize

from analyze_factor_knn_v5 import _retrieval_scores
from configs.config import config
from trainer.factor_signed_graph_stack_v21 import (
    LEXICAL_MODEL, PRODUCTION_MODEL, RETRIEVAL_MODEL, _feature_tensor,
)


def _binary_probabilities(models, matrix):
    return np.column_stack([
        model.predict_proba(matrix)[:, list(model.classes_).index(1)]
        if 1 in model.classes_ else np.zeros(matrix.shape[0])
        for model in models
    ]).astype(np.float32)


def signed_graph_stack_probabilities(
        texts, current, semantic, cpu, old_nli, prototype_nli):
    if not getattr(config, "FACTOR_USE_SIGNED_GRAPH_V21", False):
        return None, None, None
    required = (PRODUCTION_MODEL, LEXICAL_MODEL, RETRIEVAL_MODEL)
    if not all(path.exists() for path in required) or cpu is None:
        return None, None, None
    saved = joblib.load(PRODUCTION_MODEL)
    if saved.get("training_version") != "signed-label-dependency-heterogeneous-stack-v21":
        return None, None, None
    lexical_saved = joblib.load(LEXICAL_MODEL)
    lexical_matrix = lexical_saved["vectorizer"].transform(list(texts))
    lexical = _binary_probabilities(lexical_saved["models"], lexical_matrix)

    retrieval_saved = joblib.load(RETRIEVAL_MODEL)
    query = normalize(retrieval_saved["vectorizer"].transform(list(texts)))
    similarity = (query @ retrieval_saved["matrix"].T).toarray().astype(np.float32)
    retrieval = _retrieval_scores(similarity, retrieval_saved["targets"])
    components = {
        "current_v3": np.asarray(current, np.float32),
        "mentalroberta": np.asarray(semantic, np.float32),
        "tfidf": np.asarray(cpu, np.float32),
        "old_nli": np.asarray(old_nli, np.float32),
        "prototype_nli": np.asarray(prototype_nli, np.float32),
        "qwen_synthetic_lexical": lexical,
        "knn": retrieval["knn_10"],
        "positive_retrieval": retrieval["positive_max"],
    }
    if list(components) != list(saved["components"]):
        raise RuntimeError("V21 production component order mismatch")
    features = _feature_tensor(
        components, saved["graph"], saved["prevalence"],
    )
    meta = np.column_stack([
        model.predict_proba(features[:, label])[:, 1]
        for label, model in enumerate(saved["models"])
    ]).astype(np.float32)
    weight = float(saved["weight"])
    probability = (1. - weight) * components["current_v3"] + weight * meta
    return probability.astype(np.float32), np.asarray(saved["prevalence"]), float(saved["ratio"])
