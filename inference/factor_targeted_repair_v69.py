"""Apply V69 to only its two explicitly calibrated Task-2 columns."""
from __future__ import annotations

import json
import re

import numpy as np

from trainer.factor_targeted_repair_v69 import CALIBRATION


# A prevalence shrink must never delete an obvious taxonomy-positive identity
# statement.  This is deliberately narrower than the rejected V50 gazetteer:
# generic words such as "homophobic" do not prove that the author's distress
# concerns their own orientation or gender identity.
STRONG_IDENTITY = re.compile(
    r"(?:\bhate being born (?:male|female)\b"
    r"|\b(?:unhappy|uncomfortable|struggling) with my gender\b"
    r"|\b(?:i am|i['’]?m|im) (?:gay|lesbian|bisexual|bi|transgender|trans|queer)\b"
    r"|\bcoming out (?:as|to)\b"
    r"|\bwish i (?:was|were|could be) (?:a )?(?:girl|boy|woman|man)\b)",
    re.I,
)


def _rank(values):
    values = np.asarray(values, dtype=np.float32)
    order = np.argsort(values, kind="stable")
    result = np.empty(len(values), dtype=np.float32)
    result[order] = np.arange(len(values), dtype=np.float32) / max(1, len(values)-1)
    return result


def _topk(score, prevalence, ratio):
    n = len(score)
    count = max(1, min(n, int(round(n * float(prevalence) * float(ratio)))))
    selected = np.argpartition(score, n-count)[n-count:]
    result = np.zeros(n, dtype=bool); result[selected] = True
    return result


def apply_targeted_v69(probabilities, semantic_probabilities, predictions,
                       texts=None, force=False):
    if not CALIBRATION.exists():
        return predictions, False
    calibration = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    allowed = (calibration.get("experimental_adopted", False) if force
               else calibration.get("production_adopted", False))
    if not allowed:
        return predictions, False
    from configs.config import config
    original = np.asarray(predictions, dtype=bool)
    result = original.copy()
    current = np.asarray(probabilities, dtype=np.float32)
    semantic = np.asarray(semantic_probabilities, dtype=np.float32)
    prevalence = np.asarray(calibration["training_prevalence"], dtype=np.float32)
    for row in calibration["parameters"]:
        label = config.FACTOR2ID[row["label"]]
        base_rank = _rank(current[:, label])
        if row["expert"] == "semantic":
            expert_rank = _rank(semantic[:, label])
        else:
            expert_rank = base_rank
        weight = float(row["weight"])
        score = (1.0-weight)*base_rank + weight*expert_rank
        result[:, label] = _topk(score, prevalence[label], float(row["ratio"]))
        if row["label"] == "sexual orientation related issues" and texts is not None:
            protected = np.asarray([
                bool(STRONG_IDENTITY.search(str(text))) for text in texts
            ], dtype=bool)
            result[:, label] |= original[:, label] & protected
    return result, True
