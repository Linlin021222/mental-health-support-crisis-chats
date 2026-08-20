"""Tokenizer-specific caches for the standalone MentalRoBERTa factor model."""
import numpy as np
import torch
from transformers import AutoTokenizer

from configs.config import config
from preprocess.preprocess import load_train_data, load_test_data


def build_factor_cache(train=True):
    frame = load_train_data() if train else load_test_data()
    tokenizer = AutoTokenizer.from_pretrained(config.FACTOR_MODEL_NAME, use_fast=True)
    records = []
    for row in frame.itertuples(index=False):
        encoded = tokenizer(
            row.text, max_length=config.MAX_LENGTH, stride=config.STRIDE,
            truncation=True, return_overflowing_tokens=True, padding="max_length",
        )
        ids, masks = [], []
        for chunk in range(config.MAX_CHUNKS):
            if chunk < len(encoded["input_ids"]):
                ids.append(encoded["input_ids"][chunk])
                masks.append(encoded["attention_mask"][chunk])
            else:
                ids.append([tokenizer.pad_token_id] * config.MAX_LENGTH)
                masks.append([0] * config.MAX_LENGTH)
        records.append({
            "row_id": row.row_id,
            "anon_user_id": getattr(row, "anon_user_id", row.row_id),
            "text": row.text,
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
            "factor_vector": torch.tensor(
                getattr(row, "factor_vector", np.zeros(config.NUM_FACTORS)), dtype=torch.float
            ),
            "factor_counts": torch.tensor(
                getattr(row, "factor_counts", np.zeros(config.NUM_FACTORS)), dtype=torch.float
            ),
            "risk_label": int(getattr(row, "risk_label", -100)),
        })
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = config.CACHE_DIR / ("factor_train_cache.pt" if train else "factor_test_cache.pt")
    torch.save(records, path)
    print(f"Saved {len(records)} MentalRoBERTa factor records to {path}")
    return path
