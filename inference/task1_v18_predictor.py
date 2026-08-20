"""Inference helpers for the gated, full-data Task 1 V18 ensemble."""
from __future__ import annotations

import json

import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader

from configs.config import config
from datasets.collator import SuicideRiskCollator
from datasets.dataset import SuicideRiskDataset
from models.multitask_model import SuicideRiskMultiTaskModel


OUTPUT = config.OUTPUT_DIR / "task1_candidate_v18"
CALIBRATION = OUTPUT / "calibration.json"
CHECKPOINT = OUTPUT / "full_seed2_model.pt"
LEXICAL_MODEL = OUTPUT / "full_lexical_svc.joblib"
V36_OUTPUT = config.OUTPUT_DIR / "task1_oof_risk_v36"
V36_CALIBRATION = V36_OUTPUT / "calibration.json"
V36_LEXICAL_MODEL = V36_OUTPUT / "full_lexical_svc.joblib"
TRAINING_VERSION = "task1-consolidated-candidate-v18"


def load_v18_calibration():
    if not (CALIBRATION.exists() and CHECKPOINT.exists() and LEXICAL_MODEL.exists()):
        return None
    payload = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    if payload.get("training_version") != TRAINING_VERSION or not payload.get("adopted", False):
        return None
    return payload


def load_v36_risk_calibration():
    if not (V36_CALIBRATION.exists() and V36_LEXICAL_MODEL.exists()):
        return None
    payload = json.loads(V36_CALIBRATION.read_text(encoding="utf-8"))
    if not payload.get("adopted", False):
        return None
    return payload


def _softmax(values, temperature):
    values = np.asarray(values, dtype=np.float64) / float(temperature)
    values -= values.max(1, keepdims=True)
    values = np.exp(values)
    return values / values.sum(1, keepdims=True)


def v18_lexical_probabilities(texts, calibration, model_path=None):
    model_path = model_path or LEXICAL_MODEL
    saved = joblib.load(model_path)
    expected = ("task1-full-oof-risk-calibration-v36"
                if model_path == V36_LEXICAL_MODEL else TRAINING_VERSION)
    if saved.get("training_version") != expected:
        raise ValueError("Task 1 lexical artifact has the wrong training version")
    model = saved["risk_model"]
    matrix = saved["vectorizer"].transform([str(text) for text in texts])
    probability = _softmax(model.decision_function(matrix), calibration["temperature"])
    ordered = np.zeros((len(texts), config.NUM_RISK_CLASSES), dtype=np.float64)
    ordered[:, np.asarray(model.classes_, dtype=int)] = probability
    return ordered


@torch.no_grad()
def v18_seed2_evidence():
    dataset = SuicideRiskDataset(config.CACHE_DIR / "test_cache.pt")
    loader = DataLoader(
        dataset, batch_size=config.BATCH_SIZE, shuffle=False,
        collate_fn=SuicideRiskCollator(), num_workers=config.NUM_WORKERS,
    )
    device = torch.device(config.DEVICE)
    model = SuicideRiskMultiTaskModel().to(device)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device)); model.eval()
    rows = []
    for batch in loader:
        output = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        for index, row_id in enumerate(batch["row_id"]):
            rows.append({
                "row_id": str(row_id),
                "start": output["start_logits"][index].float().cpu(),
                "end": output["end_logits"][index].float().cpu(),
            })
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows


__all__ = [
    "load_v18_calibration", "load_v36_risk_calibration",
    "v18_lexical_probabilities", "v18_seed2_evidence", "V36_LEXICAL_MODEL",
]
