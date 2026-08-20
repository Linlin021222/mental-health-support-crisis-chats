"""Definition-derived boundary anchors for three semantically rare factors."""
from __future__ import annotations

import json
import re

import numpy as np

from configs.config import config


CALIBRATION = config.OUTPUT_DIR / "factor_boundary_lexicon_v50" / "calibration.json"

# These anchors encode the formal taxonomy boundary, not generic sentiment.
# A match only boosts the accepted ensemble score; final prevalence-ranked
# decoding and the neural ordering among matched posts remain intact.
PATTERNS = {
    13: [
        # Observation/exposure language is required. Generic phrases such as
        # "attempted suicide" also describe the author and were too broad.
        r"(?:watched|saw|read(?:ing)? about|video|forum).{0,100}(?:suicid|died|kill(?:ed|ing)? (?:him|her|them)self)",
    ],
    18: [
        r"\b(?:gay|lesbian|bisexual|transgender|trans girl|trans boy|queer|homophobic|homosexual|lgbtq?|gender identity|gender issues|coming out|castration)\b",
        r"\bsexual (?:orientation|identity|feelings?|partner)\b",
    ],
    23: [
        r"(?:meaning|purpose|reason|something|someone|things?) (?:in life|to live|to stay alive|to keep (?:living|going)|worth living|to continue)",
        r"(?:life (?:has|had|lost|without|is worth) meaning|meaning of life|point (?:of|in) life|worth living|reason to stay alive|reason to live|things to live for)",
        r"(?:future goals?|life goals?|dreams?|graduate|keep fighting|want to live|wanna live|choose to live)",
    ],
}
COMPILED = {label: [re.compile(pattern, re.I | re.S) for pattern in patterns]
            for label, patterns in PATTERNS.items()}
DEFAULT_BOOSTS = {13: .10, 18: .50, 23: .20}


def boundary_flags(texts):
    texts = [str(text) for text in texts]
    flags = np.zeros((len(texts), config.NUM_FACTORS), dtype=np.float32)
    for label, patterns in COMPILED.items():
        for row, text in enumerate(texts):
            flags[row, label] = float(any(pattern.search(text) for pattern in patterns))
    return flags


def apply_boundary_lexicon(texts, probabilities):
    """Apply V50 only after its development gate has adopted the policy."""
    if not CALIBRATION.exists():
        return np.asarray(probabilities, dtype=np.float32), False
    calibration = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    if not calibration.get("adopted", False):
        return np.asarray(probabilities, dtype=np.float32), False
    result = np.asarray(probabilities, dtype=np.float32).copy()
    flags = boundary_flags(texts)
    for label, boost in calibration["boosts"].items():
        label_id = int(label)
        result[:, label_id] += float(boost) * flags[:, label_id]
    return result, True
