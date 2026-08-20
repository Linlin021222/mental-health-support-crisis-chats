"""Boundary-aware evidence candidates and conservative multi-fold voting."""
from __future__ import annotations

import math
import re

import numpy as np
import torch

from configs.config import config


SPACE = re.compile(r"\s+")


def _normalise(text):
    return SPACE.sub(" ", str(text).casefold()).strip()


def decode_span_candidates(text, offsets, start_logits, end_logits, token_logits=None,
                           max_tokens=config.MAX_EVIDENCE_TOKENS, max_candidates=20):
    start = torch.sigmoid(start_logits).detach().cpu().numpy()
    end = torch.sigmoid(end_logits).detach().cpu().numpy()
    token = (None if token_logits is None
             else torch.sigmoid(token_logits).detach().cpu().numpy())
    candidates = {}
    for chunk, chunk_offsets in enumerate(offsets):
        valid_tokens = [i for i, (a, b) in enumerate(chunk_offsets) if b > a]
        starts = sorted(valid_tokens, key=lambda i: float(start[chunk, i]), reverse=True)[:16]
        ends = sorted(valid_tokens, key=lambda i: float(end[chunk, i]), reverse=True)[:16]
        starts = [i for i in starts if start[chunk, i] >= 0.25]
        ends = [i for i in ends if end[chunk, i] >= 0.25]
        for s in starts:
            for e in ends:
                if e < s or e - s + 1 > max_tokens:
                    continue
                char_start, char_end = chunk_offsets[s][0], chunk_offsets[e][1]
                phrase = text[char_start:char_end].strip()
                if not phrase:
                    continue
                boundary = math.sqrt(float(start[chunk, s] * end[chunk, e]))
                inside = boundary if token is None else float(token[chunk, s:e + 1].mean())
                length_penalty = 0.018 * max(0, e - s)
                score = 0.80 * boundary + 0.20 * inside - length_penalty
                if score < 0.25:
                    continue
                key = (int(char_start), int(char_end))
                item = {"start": key[0], "end": key[1], "phrase": phrase, "score": score}
                if key not in candidates or score > candidates[key]["score"]:
                    candidates[key] = item
    ordered = sorted(candidates.values(), key=lambda item: item["score"], reverse=True)
    selected = []
    for item in ordered:
        normal = _normalise(item["phrase"])
        if not any(normal in _normalise(old["phrase"]) or _normalise(old["phrase"]) in normal
                   for old in selected):
            selected.append(item)
        if len(selected) == max_candidates:
            break
    return selected


def candidate_phrases(candidates, maximum=config.TASK1_CV_MAX_EVIDENCE_PHRASES):
    return [item["phrase"] for item in candidates[:maximum]]


def ensemble_span_candidates(per_fold_candidates, maximum=config.TASK1_CV_MAX_EVIDENCE_PHRASES):
    groups = []
    for fold, candidates in enumerate(per_fold_candidates):
        for item in candidates[:10]:
            normal = _normalise(item["phrase"])
            group = next((group for group in groups if
                          normal in group["normal"] or group["normal"] in normal), None)
            if group is None:
                groups.append({"normal": normal, "items": [(fold, item)]})
            else:
                group["items"].append((fold, item))
                # Prefer the shorter faithful representative for containment F1.
                if len(normal.split()) < len(group["normal"].split()):
                    group["normal"] = normal
    ranked = []
    for group in groups:
        best_by_fold = {}
        for fold, item in group["items"]:
            if fold not in best_by_fold or item["score"] > best_by_fold[fold]["score"]:
                best_by_fold[fold] = item
        votes = len(best_by_fold)
        best = max(best_by_fold.values(), key=lambda item: item["score"])
        score = float(np.mean([item["score"] for item in best_by_fold.values()])) + 0.06 * votes
        ranked.append((votes, score, best))
    consensus = [item for votes, score, item in sorted(ranked, key=lambda x: (x[0], x[1]), reverse=True)
                 if votes >= 2]
    if not consensus and ranked:
        consensus = [max(ranked, key=lambda x: x[1])[2]]
    return candidate_phrases(consensus, maximum)
