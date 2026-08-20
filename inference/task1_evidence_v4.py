"""Metric-aligned evidence decoding for Task 1.

The scorer rewards short faithful phrases by containment and penalises every
extra unmatched phrase.  This module therefore keeps risk correction separate
from evidence ranking and exposes a small, calibration-driven decoder.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import torch

from baseline import ATTEMPT_CUE, BEHAVIOR_CUE, IDEATION_CUE
from configs.config import config


CALIBRATION_FILE = config.OUTPUT_DIR / "task1_evidence_v4" / "calibration.json"

# Additional explicit expressions observed in the task definition and common
# Reddit language.  Every returned match is verbatim text from the post.
EXTENDED_ATTEMPT_CUE = re.compile(
    r"(?i)(suicide attempt|attempted suicide|tried to (?:kill|end)|"
    r"survived (?:a|an|my) (?:suicide )?attempt|after my (?:suicide )?attempt)"
)
EXTENDED_BEHAVIOR_CUE = re.compile(
    r"(?i)(plan(?:ned|ning)? to (?:kill myself|die|end my life)|"
    r"going to (?:kill myself|end my life)|"
    r"(?:take|swallow)(?:ing)? (?:all|every)(?: of)? (?:my )?pills|"
    r"(?:shoot|hang|stab|cut|drown) myself|overdose(?: on)?|bleed out)"
)
EXTENDED_IDEATION_CUE = re.compile(
    r"(?i)(suicid(?:e|al)|k(?:ill)?ms|kill(?:ing)? myself|"
    r"(?:want|wanna|wanting|wish|hope)(?: to| i)? (?:fucking )?die|"
    r"wish i (?:was |were )?dead|(?:rather|better off) (?:be )?dead|"
    r"end my life|don.t want to (?:live|be alive)|no reason to live|"
    r"stop existing|sleep forever|pass away in my sleep|"
    r"not (?:wake up|see tomorrow)|die peacefully|ready to die|kill me(?: off)?)"
)


def load_evidence_calibration(path: Path = CALIBRATION_FILE):
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload.get("adopted", False):
        return None
    return payload


def correct_risk_only(text, predicted_risk):
    """Preserve the established high-confidence Indicator correction."""
    predicted_risk = int(predicted_risk)
    if predicted_risk != config.RISK_LABELS["Indicator"]:
        return predicted_risk
    if ATTEMPT_CUE.search(text):
        return config.RISK_LABELS["Attempt"]
    if BEHAVIOR_CUE.search(text):
        return config.RISK_LABELS["Behavior"]
    if IDEATION_CUE.search(text):
        return config.RISK_LABELS["Ideation"]
    return predicted_risk


def decode_model_evidence(
    text, offsets, start_logits, end_logits, threshold=0.55,
    max_tokens=8, end_policy="nearest", limit=5,
):
    """Decode verbatim spans while retaining their endpoint confidence."""
    start = torch.sigmoid(start_logits).detach().cpu().numpy()
    end = torch.sigmoid(end_logits).detach().cpu().numpy()
    candidates = []
    for chunk, chunk_offsets in enumerate(offsets):
        starts = np.flatnonzero(start[chunk] >= float(threshold))
        ends = np.flatnonzero(end[chunk] >= float(threshold))
        for token_start in starts:
            valid = [
                token_end for token_end in ends
                if token_start <= token_end <= token_start + int(max_tokens)
                and chunk_offsets[token_start][1] > chunk_offsets[token_start][0]
                and chunk_offsets[token_end][1] > chunk_offsets[token_end][0]
            ]
            if not valid:
                continue
            if end_policy == "best":
                token_end = max(
                    valid, key=lambda value: float(start[chunk, token_start] * end[chunk, value])
                )
            else:
                token_end = min(valid, key=lambda value: value - token_start)
            begin, finish = chunk_offsets[token_start][0], chunk_offsets[token_end][1]
            phrase = text[begin:finish].strip()
            if phrase:
                candidates.append((
                    float(start[chunk, token_start] * end[chunk, token_end]), phrase
                ))
    selected = []
    for _, phrase in sorted(candidates, reverse=True):
        normalized = " ".join(phrase.casefold().split())
        if not any(
            normalized in old.casefold() or old.casefold() in normalized
            for old in selected
        ):
            selected.append(phrase)
        if len(selected) == int(limit):
            break
    return selected


def _matches(patterns, text):
    return [match.group(0) for pattern in patterns for match in pattern.finditer(text)]


def cue_phrases(text, risk_id, policy):
    if policy == "none":
        return []
    legacy = {
        config.RISK_LABELS["Attempt"]: (ATTEMPT_CUE,),
        config.RISK_LABELS["Behavior"]: (BEHAVIOR_CUE,),
        config.RISK_LABELS["Ideation"]: (IDEATION_CUE,),
    }
    extended = {
        config.RISK_LABELS["Attempt"]: (EXTENDED_ATTEMPT_CUE,),
        config.RISK_LABELS["Behavior"]: (EXTENDED_BEHAVIOR_CUE,),
        config.RISK_LABELS["Ideation"]: (EXTENDED_IDEATION_CUE,),
    }
    if policy.startswith("current"):
        return _matches((ATTEMPT_CUE, BEHAVIOR_CUE, IDEATION_CUE), text)
    patterns = list(legacy.get(int(risk_id), ()))
    if "extended" in policy:
        patterns += list(extended.get(int(risk_id), ()))
    if "hierarchical" in policy:
        if int(risk_id) == config.RISK_LABELS["Attempt"]:
            patterns += list(legacy[config.RISK_LABELS["Behavior"]])
            patterns += list(legacy[config.RISK_LABELS["Ideation"]])
            if "extended" in policy:
                patterns += list(extended[config.RISK_LABELS["Behavior"]])
                patterns += list(extended[config.RISK_LABELS["Ideation"]])
        elif int(risk_id) == config.RISK_LABELS["Behavior"]:
            patterns += list(legacy[config.RISK_LABELS["Ideation"]])
            if "extended" in policy:
                patterns += list(extended[config.RISK_LABELS["Ideation"]])
    return _matches(tuple(patterns), text)


def apply_evidence_policy(text, risk_id, model_phrases, policy="current_first", topk=5):
    """Fuse model spans and short explicit cues without emitting duplicates."""
    if int(risk_id) == config.RISK_LABELS["Indicator"]:
        return []
    cues = cue_phrases(text, risk_id, policy)
    phrases = list(model_phrases) + cues if policy.endswith("model_first") else cues + list(model_phrases)
    # A semicolon is the submission delimiter.  Split it before top-k so one
    # model span cannot silently become multiple predictions after writing CSV.
    phrases = [part.strip() for phrase in phrases for part in str(phrase).split(";") if part.strip()]
    selected = []
    for phrase in phrases:
        normalized = " ".join(str(phrase).casefold().split())
        if phrase and not any(
            normalized in old.casefold() or old.casefold() in normalized
            for old in selected
        ):
            selected.append(str(phrase))
        if len(selected) == int(topk):
            break
    return selected
