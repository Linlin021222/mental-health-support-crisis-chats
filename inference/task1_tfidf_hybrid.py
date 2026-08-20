"""Gated TF-IDF risk probabilities for the accepted Task 1 hybrid."""
import json

import joblib
import numpy as np

from baseline import MODEL_FILE
from configs.config import config


CALIBRATION_FILE = config.OUTPUT_DIR / "task1_tfidf_hybrid" / "calibration.json"
TRAINING_VERSION = "tfidf-risk-hybrid-v1"


def _softmax(values, temperature):
    values = np.asarray(values, dtype=np.float32) / float(temperature)
    values -= values.max(axis=1, keepdims=True)
    values = np.exp(values)
    return values / values.sum(axis=1, keepdims=True)


def task1_tfidf_probabilities(texts):
    if not CALIBRATION_FILE.exists() or not MODEL_FILE.exists():
        return None, None
    calibration = json.loads(CALIBRATION_FILE.read_text(encoding="utf-8"))
    if (calibration.get("training_version") != TRAINING_VERSION
            or not calibration.get("adopted", False)):
        return None, calibration
    if calibration.get("lexical_model") != "svc_temperature_0.5":
        print("Task 1 TF-IDF calibration is unsupported; using stable risk model")
        return None, calibration
    saved = joblib.load(MODEL_FILE)
    vectorizer, model = saved.get("vectorizer"), saved.get("risk_model")
    if vectorizer is None or model is None or not hasattr(model, "decision_function"):
        print("Task 1 TF-IDF model incomplete; using stable risk model")
        return None, calibration
    matrix = vectorizer.transform([str(text) for text in texts])
    decision = model.decision_function(matrix)
    probabilities = _softmax(decision, calibration["temperature"])
    classes = np.asarray(model.classes_, dtype=int)
    ordered = np.zeros((len(texts), config.NUM_RISK_CLASSES), dtype=np.float32)
    ordered[:, classes] = probabilities
    return ordered, calibration


__all__ = ["task1_tfidf_probabilities"]
