"""Inference helper for the optional five-fold MHLAT-v4 Task 2 expert."""
import json

import numpy as np
import torch
from torch.utils.data import DataLoader

from configs.config import config
from datasets.factor_cache_builder import build_factor_cache
from models.factor_mhlat_model import MentalRobertaMHLATModel
from trainer.factor_mhlat_v4 import CALIBRATION_FILE, OUTPUT_DIR, TRAINING_VERSION
from trainer.factor_train import FactorDataset, _collate


CHECKPOINTS = [OUTPUT_DIR / f"fold{fold}_model.pt" for fold in range(config.N_FOLDS)]
TEST_CACHE = OUTPUT_DIR / "test_probabilities.npz"


def _signature():
    return json.dumps({
        "training_version": TRAINING_VERSION,
        "checkpoints": [
            [path.name, int(path.stat().st_size), int(path.stat().st_mtime_ns)]
            for path in CHECKPOINTS
        ],
    }, sort_keys=True)


@torch.no_grad()
def mhlat_factor_probabilities(force=False):
    if not config.FACTOR_USE_MHLAT_V4 or not CALIBRATION_FILE.exists():
        return None, None, None
    calibration = json.loads(CALIBRATION_FILE.read_text(encoding="utf-8"))
    if calibration.get("training_version") != TRAINING_VERSION:
        print("MHLAT-v4 Task 2 artefacts are stale; using accepted v3")
        return None, None, calibration
    if not calibration.get("adopted", False):
        print("MHLAT-v4 Task 2 expert was rejected by nested user-disjoint validation")
        return None, None, calibration
    missing = [path for path in CHECKPOINTS if not path.exists()]
    if missing:
        print(f"MHLAT-v4 disabled; missing checkpoints: {missing}")
        return None, None, calibration

    cache = build_factor_cache(train=False)
    dataset = FactorDataset(cache)
    row_ids = np.asarray([str(x["row_id"]) for x in dataset.data])
    signature = _signature()
    if TEST_CACHE.exists() and not force:
        saved = np.load(TEST_CACHE)
        if (saved["probabilities"].shape == (len(dataset), config.NUM_FACTORS)
                and np.array_equal(saved["row_ids"].astype(str), row_ids)
                and str(saved["checkpoint_signature"]) == signature):
            return row_ids, saved["probabilities"].astype(np.float32), calibration

    loader = DataLoader(
        dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=_collate,
        num_workers=config.NUM_WORKERS,
    )
    device = torch.device(config.DEVICE)
    probabilities = np.zeros((len(dataset), config.NUM_FACTORS), dtype=np.float32)
    for fold, checkpoint in enumerate(CHECKPOINTS):
        model = MentalRobertaMHLATModel(initialise_labels=False).to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        model.eval(); chunks = []
        for batch in loader:
            logits = model(
                batch["input_ids"].to(device), batch["attention_mask"].to(device)
            )
            chunks.append(torch.sigmoid(logits).cpu().numpy())
        probabilities += np.vstack(chunks) / len(CHECKPOINTS)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"MHLAT-v4 factor fold {fold + 1}/{len(CHECKPOINTS)} complete")
    np.savez_compressed(
        TEST_CACHE, probabilities=probabilities, row_ids=row_ids,
        checkpoint_signature=signature,
    )
    return row_ids, probabilities, calibration


__all__ = ["mhlat_factor_probabilities"]
