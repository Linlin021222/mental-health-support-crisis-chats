"""Conservative polarity-aware correction for Task 1.

The released annotations contain a small but important Indicator subtype:
explicit suicide language under negation (for example, ``don't wanna die``).
The ordinary cue rules see only the suicidal core and can promote these posts
to Ideation.  This module changes a prediction only when the negated phrase is
the sole explicit suicide statement in the post.  The returned evidence is
the shortest verbatim suicidal core so it remains robust to apostrophe
normalisation in the workbook's gold evidence.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

from configs.config import config


CALIBRATION = config.OUTPUT_DIR / "task1_polarity_v63" / "calibration.json"

NEGATED_SUICIDE = re.compile(
    r"(?i)(?:don(?:\W)?t|do\s+not|never)\s+(?:really\s+)?"
    r"(?P<core>(?:(?:wanna|want\s+to|wish\s+to)\s+)?"
    r"(?:die|kill\s+myself|kms|end\s+(?:it|my\s+life)))"
)

# Evaluated after removing NEGATED_SUICIDE.  Word boundaries are deliberate:
# substring matching such as ``cut`` in an unrelated word is unsafe here.
RESIDUAL_EXPLICIT_SUICIDE = re.compile(
    r"(?i)\bsuicid(?:e|al)\b|"
    r"\b(?:hope|wish|want|wanna|need)\s+(?:i\s+|to\s+)?"
    r"(?:die|be\s+dead)\b|"
    r"\bno\s+longer\s+wish\s+to\s+(?:live|continue)\b|"
    r"\b(?:plan|going|intend|ready)\s+to\s+(?:die|kill|end)\b|"
    r"\b(?:tried|attempted|overdose|hang|shoot|jump|cut)\b"
)


@lru_cache(maxsize=1)
def load_polarity_calibration(path: Path = CALIBRATION):
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if payload.get("adopted", False) else None


def polarity_candidate(text: str, risk_id: int):
    """Return ``(Indicator, [verbatim core])`` or ``None``.

    Attempt is never changed.  Behavior is eligible only because a false
    Behavior prediction can arise from the word ``kill`` inside a negation;
    any residual plan/method cue blocks the correction.
    """
    if int(risk_id) not in (
        config.RISK_LABELS["Ideation"], config.RISK_LABELS["Behavior"],
    ):
        return None
    match = NEGATED_SUICIDE.search(str(text))
    if match is None:
        return None
    residual = NEGATED_SUICIDE.sub(" ", str(text))
    if RESIDUAL_EXPLICIT_SUICIDE.search(residual):
        return None
    return config.RISK_LABELS["Indicator"], [match.group("core")]


def apply_polarity_correction(text: str, risk_id: int, evidence: list[str]):
    calibration = load_polarity_calibration()
    if calibration is None:
        return int(risk_id), list(evidence), False
    candidate = polarity_candidate(text, risk_id)
    if candidate is None:
        return int(risk_id), list(evidence), False
    new_risk, new_evidence = candidate
    return int(new_risk), list(new_evidence), True
