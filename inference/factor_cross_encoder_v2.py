"""Inference for the optional five-fold prototype-bank Task 2 refinement."""
import json

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from configs.config import config
from trainer.factor_cross_encoder_v2 import (
    CALIBRATION_FILE, OUTPUT_DIR, TRAINING_VERSION, _predict,
)


CHECKPOINTS = [OUTPUT_DIR / f"fold{fold}_model.pt" for fold in range(config.N_FOLDS)]
TEST_CACHE = OUTPUT_DIR / "test_probabilities.npz"


def _signature():
    return json.dumps({
        "training_version": TRAINING_VERSION,
        "max_chunks": int(config.FACTOR_NLI_MAX_CHUNKS),
        "checkpoints": [
            [path.name, int(path.stat().st_size), int(path.stat().st_mtime_ns)]
            for path in CHECKPOINTS
        ],
    }, sort_keys=True)


@torch.no_grad()
def prototype_cross_encoder_probabilities(texts, row_ids, force=False):
    if not config.FACTOR_USE_CROSS_ENCODER_V2 or not CALIBRATION_FILE.exists():
        return None, None
    calibration = json.loads(CALIBRATION_FILE.read_text(encoding="utf-8"))
    if calibration.get("training_version") != TRAINING_VERSION:
        print("Prototype Task 2 artefacts are stale; using stable model")
        return None, calibration
    if not calibration.get("adopted", False):
        print("Prototype Task 2 refinement was not adopted by OOF validation")
        return None, calibration
    missing = [path for path in CHECKPOINTS if not path.exists()]
    if missing:
        print(f"Prototype Task 2 refinement disabled; missing checkpoints: {missing}")
        return None, calibration
    texts = [str(text) for text in texts]
    row_ids = np.asarray(row_ids).astype(str)
    signature = _signature()
    if TEST_CACHE.exists() and not force:
        saved = np.load(TEST_CACHE)
        if (saved["probabilities"].shape == (len(texts), config.NUM_FACTORS)
                and np.array_equal(saved["row_ids"].astype(str), row_ids)
                and str(saved["checkpoint_signature"]) == signature):
            return saved["probabilities"].astype(np.float32), calibration
    tokenizer = AutoTokenizer.from_pretrained(
        config.FACTOR_NLI_MODEL_NAME, use_fast=True, local_files_only=True
    )
    device = torch.device(config.DEVICE)
    total = np.zeros((len(texts), config.NUM_FACTORS), dtype=np.float32)
    for fold, checkpoint in enumerate(CHECKPOINTS):
        model = AutoModelForSequenceClassification.from_pretrained(
            config.FACTOR_NLI_MODEL_NAME, dtype=torch.float32, local_files_only=True,
        ).to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        total += _predict(model, tokenizer, texts, device) / len(CHECKPOINTS)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"Prototype cross-encoder fold {fold + 1}/{len(CHECKPOINTS)} complete")
    TEST_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        TEST_CACHE, probabilities=total, row_ids=row_ids,
        checkpoint_signature=signature,
    )
    return total, calibration
