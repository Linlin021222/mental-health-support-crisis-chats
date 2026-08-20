"""Inference for the leak-free five-fold Task 1 ensemble.

The ensemble is deliberately opt-in: it is used only when cross-fitted OOF
validation marked the calibration as adopted.  Missing or rejected artefacts
therefore leave the established submission pipeline unchanged.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from configs.config import config
from datasets.collator import SuicideRiskCollator
from datasets.dataset import SuicideRiskDataset
from inference.task1_span_decoder import decode_span_candidates, ensemble_span_candidates
from models.task1_joint_model import Task1JointModel, ordinal_class_probabilities


OUTPUT_DIR = config.OUTPUT_DIR / "task1_cv"
CALIBRATION_FILE = OUTPUT_DIR / "calibration.json"
CACHE_FILE = OUTPUT_DIR / "test_predictions.npz"


def _artefact_signature(checkpoints, calibration):
    parts = [str(path.stat().st_mtime_ns) for path in checkpoints]
    parts.append(json.dumps(calibration, sort_keys=True))
    return "|".join(parts)


def _load_cache(row_ids, signature):
    if not CACHE_FILE.exists():
        return None
    try:
        saved = np.load(CACHE_FILE, allow_pickle=True)
        if str(saved["signature"].item()) != signature:
            return None
        cached_ids = saved["row_ids"].tolist()
        if list(map(str, cached_ids)) != list(map(str, row_ids)):
            return None
        return saved["risk_probabilities"], saved["evidence"].tolist()
    except (OSError, KeyError, ValueError):
        return None


@torch.no_grad()
def task1_cv_predictions():
    """Return ``(row_ids, risk probabilities, evidence, calibration)``.

    ``(None, None, None, None)`` means that the cross-fitted gate did not
    approve the ensemble or its artefacts are incomplete.
    """
    if not config.TASK1_USE_CV_ENSEMBLE or not CALIBRATION_FILE.exists():
        return None, None, None, None
    calibration = json.loads(CALIBRATION_FILE.read_text(encoding="utf-8"))
    if not calibration.get("adopted", False):
        print("Task 1 CV ensemble was rejected by cross-fitted validation; using stable model")
        return None, None, None, calibration
    checkpoints = [OUTPUT_DIR / f"fold{fold}_model.pt" for fold in range(config.N_FOLDS)]
    if not all(path.exists() for path in checkpoints):
        print("Task 1 CV ensemble incomplete; using stable model")
        return None, None, None, calibration

    dataset = SuicideRiskDataset(config.CACHE_DIR / "test_cache.pt")
    row_ids = [item["row_id"] for item in dataset.data]
    signature = _artefact_signature(checkpoints, calibration)
    cached = _load_cache(row_ids, signature)
    if cached is not None:
        return row_ids, cached[0], cached[1], calibration

    loader = DataLoader(
        dataset, batch_size=config.BATCH_SIZE, shuffle=False,
        collate_fn=SuicideRiskCollator(), num_workers=config.NUM_WORKERS,
        pin_memory=config.DEVICE == "cuda",
    )
    device = torch.device(config.DEVICE)
    fold_probabilities = []
    per_post_candidates = [[] for _ in row_ids]
    ordinal_weight = float(calibration["ordinal_weight"])
    for fold, checkpoint in enumerate(checkpoints):
        model = Task1JointModel().to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        model.eval(); probabilities = []; cursor = 0
        for batch in loader:
            output = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            standard = torch.softmax(output["risk_logits"], -1)
            ordinal = ordinal_class_probabilities(output["ordinal_logits"])
            probabilities.append(
                ((1.0 - ordinal_weight) * standard + ordinal_weight * ordinal).cpu().numpy()
            )
            for local in range(len(batch["row_id"])):
                per_post_candidates[cursor + local].append(decode_span_candidates(
                    batch["texts"][local], batch["offset_mappings"][local],
                    output["start_logits"][local], output["end_logits"][local],
                    output["token_logits"][local],
                ))
            cursor += len(batch["row_id"])
        fold_probabilities.append(np.vstack(probabilities))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"Task 1 CV inference fold {fold + 1}/{len(checkpoints)} complete")

    risk_probabilities = np.mean(fold_probabilities, axis=0)
    maximum = int(calibration.get("topk", config.TASK1_CV_MAX_EVIDENCE_PHRASES))
    evidence = [ensemble_span_candidates(items, maximum=maximum)
                for items in per_post_candidates]
    evidence_array = np.empty(len(evidence), dtype=object); evidence_array[:] = evidence
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        CACHE_FILE, signature=signature, row_ids=np.asarray(row_ids),
        risk_probabilities=risk_probabilities, evidence=evidence_array,
    )
    return row_ids, risk_probabilities, evidence, calibration


__all__ = ["task1_cv_predictions"]
