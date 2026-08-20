"""Build reproducible tensor caches, retaining offsets needed at inference."""
import os
import re
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer
from configs.config import config
from preprocess.preprocess import load_train_data, load_test_data


def _all_occurrences(text, phrase):
    """Case-insensitive matching on original-character offsets.

    The workbook evidence occasionally normalises apostrophes/whitespace or
    uses an ellipsis to join two verbatim fragments.  Exact-only matching used
    to silently discard those annotations.  Token matching below recovers the
    unambiguous cases while still returning offsets into the untouched post.
    """
    if not phrase:
        return []
    exact = [(m.start(), m.end()) for m in re.finditer(
        re.escape(phrase), text, flags=re.IGNORECASE)]
    if exact:
        return exact

    # An annotation such as ``gun ... blow my face off`` denotes two source
    # fragments, not one long evidence span.  Supervising each fragment also
    # agrees with the official containment-based phrase metric.
    fragments = [part.strip() for part in re.split(r"(?:\.{3,}|…+)", phrase)
                 if part.strip()]
    if len(fragments) > 1:
        spans = []
        for fragment in fragments:
            spans.extend(_all_occurrences(text, fragment))
        return sorted(set(spans))

    token_pattern = re.compile(r"[^\W_]+", flags=re.UNICODE)
    source_tokens = [(match.group(0).casefold(), match.start(), match.end())
                     for match in token_pattern.finditer(text)]
    target = [match.group(0).casefold()
              for match in token_pattern.finditer(phrase)]
    if not target:
        return []
    width = len(target); spans = []
    for start in range(len(source_tokens) - width + 1):
        if [item[0] for item in source_tokens[start:start + width]] == target:
            spans.append((source_tokens[start][1], source_tokens[start + width - 1][2]))
    return spans


def build_cache(train=True):
    """Create ``train_cache.pt`` or ``test_cache.pt`` from the Excel files.

    Each record has a fixed number of sliding-window chunks.  Crucially, the
    original post and each chunk's offset mapping are saved for faithful phrase
    reconstruction; they must never be inferred from token ids.
    """
    frame = load_train_data() if train else load_test_data()
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME, use_fast=True, local_files_only=True)
    records = []
    for row in frame.itertuples(index=False):
        text = row.text
        encoded = tokenizer(text, max_length=config.MAX_LENGTH, stride=config.STRIDE,
                            truncation=True, return_overflowing_tokens=True,
                            return_offsets_mapping=True, padding="max_length")
        n = min(len(encoded["input_ids"]), config.MAX_CHUNKS)
        ids, masks, starts, ends, token_labels, offsets = [], [], [], [], [], []
        evidence = getattr(row, "evidence", [])
        # Locate every annotation once in the original text.  This works for
        # repeated phrases and preserves the evaluation's case-insensitive rule.
        gold_spans = [span for phrase in evidence for span in _all_occurrences(text, phrase)]
        for i in range(config.MAX_CHUNKS):
            if i < n:
                chunk_offsets = [tuple(pair) for pair in encoded["offset_mapping"][i]]
                input_ids, attention = encoded["input_ids"][i], encoded["attention_mask"][i]
                s = np.zeros(config.MAX_LENGTH, dtype=np.float32)
                e = np.zeros(config.MAX_LENGTH, dtype=np.float32)
                t = np.zeros(config.MAX_LENGTH, dtype=np.float32)
                for cs, ce in gold_spans:
                    token_ids = [j for j, (a, b) in enumerate(chunk_offsets) if b > cs and a < ce and b > a]
                    if token_ids:
                        s[token_ids[0]] = 1.0
                        e[token_ids[-1]] = 1.0
                        t[token_ids] = 1.0
            else:
                input_ids = [tokenizer.pad_token_id] * config.MAX_LENGTH
                attention = [0] * config.MAX_LENGTH
                chunk_offsets = [(0, 0)] * config.MAX_LENGTH
                s = e = t = np.zeros(config.MAX_LENGTH, dtype=np.float32)
            ids.append(input_ids); masks.append(attention); starts.append(s); ends.append(e)
            token_labels.append(t); offsets.append(chunk_offsets)
        records.append({
            "row_id": row.row_id, "anon_user_id": getattr(row, "anon_user_id", row.row_id),
            "text": text, "offset_mapping": offsets,
            "evidence": evidence,
            "input_ids": torch.tensor(ids, dtype=torch.long), "attention_mask": torch.tensor(masks, dtype=torch.long),
            "start_labels": torch.tensor(np.asarray(starts)), "end_labels": torch.tensor(np.asarray(ends)),
            "token_labels": torch.tensor(np.asarray(token_labels)),
            "risk_label": torch.tensor(getattr(row, "risk_label", -100), dtype=torch.long),
            "factor_vector": torch.tensor(getattr(row, "factor_vector", np.zeros(config.NUM_FACTORS)), dtype=torch.float),
        })
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = config.CACHE_DIR / ("train_cache.pt" if train else "test_cache.pt")
    torch.save(records, path)
    print(f"Saved {len(records)} records to {path}")


if __name__ == "__main__":
    build_cache(True)
    build_cache(False)
