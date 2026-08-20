"""Measure evidence coverage of leak-free long-post window-selection rules."""
import re
import numpy as np
from transformers import AutoTokenizer

from configs.config import config
from datasets.cache_builder import _all_occurrences
from preprocess.preprocess import load_train_data


RISK_CUES = re.compile(
    r"(?i)\b(suicid(?:e|al)|kill(?:ing)? myself|end my life|want(?:ing)? to die|"
    r"wish(?:ing)? (?:i|to) .*?dead|better off dead|not wake up|sleep forever|"
    r"overdos(?:e|ing)|hang(?:ing)? myself|shoot myself|cut my wrists?|"
    r"attempt(?:ed|ing)?|pills?|rope|gun|knife|razor|jump(?:ing)?|drown(?:ing)?)\b"
)


def _selected(strategy, scores, n):
    if n <= config.MAX_CHUNKS:
        return list(range(n))
    if strategy == "first":
        return list(range(config.MAX_CHUNKS))
    if strategy == "first_last":
        return [0, n - 1]
    if strategy == "first_cue":
        best = max(range(1, n), key=lambda i: (scores[i], -i))
        return sorted({0, best})
    if strategy == "cue":
        return sorted(sorted(range(n), key=lambda i: (scores[i], -i), reverse=True)[:2])
    raise ValueError(strategy)


def main():
    frame = load_train_data()
    tokenizer = AutoTokenizer.from_pretrained(
        config.MODEL_NAME, use_fast=True, local_files_only=True
    )
    result = {name: {"posts": 0, "phrases": 0, "full_phrases": 0}
              for name in ("first", "first_last", "first_cue", "cue")}
    gold_posts = 0
    gold_phrases = 0
    chunks = []
    for row in frame.itertuples(index=False):
        spans = [span for phrase in row.evidence
                 for span in _all_occurrences(row.text, phrase)]
        if not spans:
            continue
        gold_posts += 1
        gold_phrases += len(spans)
        encoded = tokenizer(
            row.text, max_length=config.MAX_LENGTH, stride=config.STRIDE,
            truncation=True, return_overflowing_tokens=True,
            return_offsets_mapping=True,
        )
        n = len(encoded["input_ids"])
        chunks.append(n)
        bounds, scores = [], []
        for offsets in encoded["offset_mapping"]:
            real = [(a, b) for a, b in offsets if b > a]
            start, end = real[0][0], real[-1][1]
            bounds.append((start, end))
            scores.append(len(RISK_CUES.findall(row.text[start:end])))
        for name in result:
            selected = _selected(name, scores, n)
            windows = [bounds[i] for i in selected]
            overlaps = [any(b > gs and a < ge for a, b in windows) for gs, ge in spans]
            full = [any(a <= gs and b >= ge for a, b in windows) for gs, ge in spans]
            result[name]["posts"] += int(any(overlaps))
            result[name]["phrases"] += sum(overlaps)
            result[name]["full_phrases"] += sum(full)
    print(f"gold posts={gold_posts} phrase occurrences={gold_phrases}")
    print("chunk p50/p90/max", np.percentile(chunks, [50, 90, 100]).tolist())
    for name, values in result.items():
        print(name, values, {
            "post_recall": values["posts"] / gold_posts,
            "phrase_overlap_recall": values["phrases"] / gold_phrases,
            "phrase_full_recall": values["full_phrases"] / gold_phrases,
        })


if __name__ == "__main__":
    main()
