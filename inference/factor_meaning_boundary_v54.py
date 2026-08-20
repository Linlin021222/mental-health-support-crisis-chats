"""Graded high-precision meaning-in-life boundary evidence for inference."""
from __future__ import annotations

import json
import re

import numpy as np

from configs.config import config


CALIBRATION = config.OUTPUT_DIR / "factor_meaning_boundary_v54" / "calibration.json"
LABEL = 23
STRONG_PATTERN = re.compile(
    r"(?:\b(?:meaning|purpose) (?:of|in) (?:my |this )?life\b"
    r"|\blife.{0,25}\b(?:meaning|purpose)\b"
    r"|\b(?:reason|something|someone|things?) "
    r"(?:to live|to stay alive|to keep (?:living|going)|worth living|to continue)\b"
    r"|\b(?:worth living|reason to stay alive|reason to live|things to live for"
    r"|point (?:of|in) life|life worth living|life has to offer)\b"
    r"|\bwhat does life (?:even )?have to offer\b)",
    re.I | re.S,
)


def strong_meaning_flags(texts):
    return np.asarray([bool(STRONG_PATTERN.search(str(text))) for text in texts],
                      dtype=np.float32)


def apply_meaning_boundary_v54(texts, probabilities):
    result = np.asarray(probabilities, dtype=np.float32).copy()
    if not CALIBRATION.exists():
        return result, False
    calibration = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    if not calibration.get("adopted", False):
        return result, False
    from inference.factor_boundary_lexicon_v50 import boundary_flags
    broad = boundary_flags(texts)[:, LABEL]
    strong = strong_meaning_flags(texts)
    # V50 already adds 0.20 to broad evidence; these are incremental boosts.
    result[:, LABEL] += float(calibration["additional_broad_boost"]) * broad
    result[:, LABEL] += float(calibration["strong_boost"]) * strong
    return result, True
