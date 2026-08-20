"""Five-fold shared text/label cross-encoder inference for Task 2."""
import json
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from configs.config import config
from inference.factor_nli import _entailment_index


CROSS_DIR = config.OUTPUT_DIR / "factor_cross_encoder"
CROSS_CHECKPOINTS = [CROSS_DIR / f"fold{i}_model.pt" for i in range(config.N_FOLDS)]
CROSS_TEST_CACHE = CROSS_DIR / "test_probabilities.npz"


def _checkpoint_signature():
    return json.dumps([
        [path.name, int(path.stat().st_size), int(path.stat().st_mtime_ns)]
        for path in CROSS_CHECKPOINTS
    ])


@torch.no_grad()
def _model_probabilities(model, tokenizer, texts, device):
    model.eval()
    entailment = _entailment_index(model)
    result = np.zeros((len(texts), config.NUM_FACTORS), dtype=np.float32)
    batch_size = config.FACTOR_CROSS_ENCODER_BATCH_SIZE * 2
    for label, hypothesis in enumerate(tqdm(
        config.FACTOR_NLI_HYPOTHESES, desc="cross-encoder labels"
    )):
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            encoded = tokenizer(
                batch, [hypothesis] * len(batch), padding=True,
                truncation="only_first", max_length=config.FACTOR_NLI_MAX_LENGTH,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(**encoded).logits.float()
            result[start:start + len(batch), label] = (
                torch.softmax(logits, dim=-1)[:, entailment].cpu().numpy()
            )
    return result


@torch.no_grad()
def cross_encoder_probabilities(texts, row_ids, force=False):
    """Average five user-disjoint classifiers; cache only an exact test match."""
    if not config.FACTOR_USE_CROSS_ENCODER:
        return None
    missing = [path for path in CROSS_CHECKPOINTS if not path.exists()]
    if missing:
        print(f"Cross-encoder disabled; missing checkpoints: {missing}")
        return None
    texts = [str(x) for x in texts]
    row_ids = np.asarray(row_ids).astype(str)
    signature = _checkpoint_signature()
    if CROSS_TEST_CACHE.exists() and not force:
        saved = np.load(CROSS_TEST_CACHE)
        if (saved["probabilities"].shape == (len(texts), config.NUM_FACTORS)
                and np.array_equal(saved["row_ids"].astype(str), row_ids)
                and str(saved["checkpoint_signature"]) == signature):
            print(f"Loaded cached five-fold cross-encoder probabilities: {CROSS_TEST_CACHE}")
            return saved["probabilities"].astype(np.float32)

    tokenizer = AutoTokenizer.from_pretrained(
        config.FACTOR_NLI_MODEL_NAME, use_fast=True, local_files_only=True
    )
    device = torch.device(config.DEVICE)
    total = np.zeros((len(texts), config.NUM_FACTORS), dtype=np.float32)
    for fold, checkpoint in enumerate(CROSS_CHECKPOINTS):
        model = AutoModelForSequenceClassification.from_pretrained(
            config.FACTOR_NLI_MODEL_NAME, dtype=torch.float32, local_files_only=True,
        ).to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        total += _model_probabilities(model, tokenizer, texts, device) / len(CROSS_CHECKPOINTS)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"Cross-encoder fold {fold + 1}/{len(CROSS_CHECKPOINTS)} complete")
    CROSS_TEST_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        CROSS_TEST_CACHE, probabilities=total, row_ids=row_ids,
        checkpoint_signature=signature,
    )
    print(f"Saved five-fold cross-encoder probabilities: {CROSS_TEST_CACHE}")
    return total

