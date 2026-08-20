"""Paper-definition NLI expert for rare suicide-factor labels.

This model is deliberately independent from the supervised Task 2 models.  It
turns each factor definition into a natural-language hypothesis and asks an
NLI model whether a post entails it.  That gives tail labels a useful signal
even when only a handful of labelled training examples exist.
"""
from pathlib import Path
import json
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from configs.config import config


NLI_DIR = config.OUTPUT_DIR / "factor_nli"
TRAIN_NLI_FILE = NLI_DIR / "train_probabilities.npz"
TEST_NLI_FILE = NLI_DIR / "test_probabilities.npz"
NLI_CALIBRATION_FILE = NLI_DIR / "calibration.json"


def _word_chunks(text, words_per_chunk=300, overlap=60):
    """Cover the beginning, middle and end of long posts without huge batches."""
    words = str(text).split()
    if not words:
        return [""]
    if len(words) <= words_per_chunk:
        return [" ".join(words)]
    step = max(1, words_per_chunk - overlap)
    starts = list(range(0, len(words), step))
    windows = [" ".join(words[s:s + words_per_chunk]) for s in starts]
    windows = [x for x in windows if x]
    maximum = int(config.FACTOR_NLI_MAX_CHUNKS)
    if len(windows) <= maximum:
        return windows
    # Evenly-spaced selection avoids the old first-chunks-only blind spot.
    selected = np.linspace(0, len(windows) - 1, maximum).round().astype(int)
    return [windows[int(i)] for i in np.unique(selected)]


def _entailment_index(model):
    mapping = {str(k).casefold(): int(v) for k, v in model.config.label2id.items()}
    for name, index in mapping.items():
        if "entail" in name and "not" not in name:
            return index
    # The selected MoritzLaurer binary checkpoint documents entailment as
    # label 0.  Fail loudly for an incompatible replacement model.
    if model.config.num_labels == 2:
        return 0
    raise ValueError(f"Cannot identify entailment label: {model.config.label2id}")


@torch.no_grad()
def compute_nli_probabilities(texts, row_ids=None, cache_path=None, force=False):
    """Return [posts, 24] entailment probabilities with max-over-chunks MIL."""
    texts = [str(x) for x in texts]
    row_ids = np.asarray(list(range(len(texts))) if row_ids is None else row_ids).astype(str)
    cache_path = Path(cache_path) if cache_path is not None else None
    if cache_path is not None and cache_path.exists() and not force:
        saved = np.load(cache_path)
        if (saved["probabilities"].shape == (len(texts), config.NUM_FACTORS)
                and np.array_equal(saved["row_ids"].astype(str), row_ids)):
            print(f"Loaded cached factor NLI probabilities: {cache_path}")
            return saved["probabilities"].astype(np.float32)

    device = torch.device(config.DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(config.FACTOR_NLI_MODEL_NAME, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.FACTOR_NLI_MODEL_NAME, dtype=torch.float32,
    ).to(device)
    model.eval()
    entailment = _entailment_index(model)
    print(f"Factor NLI label map: {model.config.label2id}; entailment={entailment}")

    chunks = [_word_chunks(text) for text in texts]
    probabilities = np.zeros((len(texts), config.NUM_FACTORS), dtype=np.float32)
    batch_size = int(config.FACTOR_NLI_BATCH_SIZE)
    for label_id, hypothesis in enumerate(tqdm(
        config.FACTOR_NLI_HYPOTHESES, desc="factor NLI labels"
    )):
        owners, premises = [], []
        for owner, post_chunks in enumerate(chunks):
            owners.extend([owner] * len(post_chunks))
            premises.extend(post_chunks)
        for start in range(0, len(premises), batch_size):
            stop = min(start + batch_size, len(premises))
            encoded = tokenizer(
                premises[start:stop], [hypothesis] * (stop - start),
                padding=True, truncation="only_first",
                max_length=config.FACTOR_NLI_MAX_LENGTH, return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(**encoded).logits.float()
            if logits.shape[-1] == 1:
                scores = torch.sigmoid(logits[:, 0])
            else:
                scores = torch.softmax(logits, dim=-1)[:, entailment]
            values = scores.cpu().numpy()
            for owner, value in zip(owners[start:stop], values):
                probabilities[owner, label_id] = max(
                    probabilities[owner, label_id], float(value)
                )

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, probabilities=probabilities, row_ids=row_ids)
        print(f"Saved factor NLI probabilities: {cache_path}")
    return probabilities


def _rank_decode(probabilities, prevalence, ratios):
    probabilities = np.asarray(probabilities, dtype=np.float32)
    prevalence = np.asarray(prevalence, dtype=np.float32)
    ratios = np.broadcast_to(np.asarray(ratios, dtype=np.float32), prevalence.shape)
    prediction = np.zeros_like(probabilities, dtype=bool)
    n = len(probabilities)
    for label in range(probabilities.shape[1]):
        count = max(1, int(round(n * float(prevalence[label]) * float(ratios[label]))))
        count = min(n, count)
        chosen = np.argpartition(probabilities[:, label], n - count)[n - count:]
        prediction[chosen, label] = True
    return prediction


def apply_nli_calibration(base_probabilities, nli_probabilities, calibration):
    """Blend per label and decode with prevalence ranks selected by cross-fit."""
    base = np.asarray(base_probabilities, dtype=np.float32)
    nli = np.asarray(nli_probabilities, dtype=np.float32)
    weights = np.asarray(calibration["nli_weights"], dtype=np.float32)
    ratios = np.asarray(calibration["prevalence_ratios"], dtype=np.float32)
    prevalence = np.asarray(calibration["training_prevalence"], dtype=np.float32)
    mixed = (1.0 - weights[None, :]) * base + weights[None, :] * nli
    return _rank_decode(mixed, prevalence, ratios), mixed


def load_nli_calibration(path=NLI_CALIBRATION_FILE):
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
