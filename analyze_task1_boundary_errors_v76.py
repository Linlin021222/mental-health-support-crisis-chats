"""Structured error audit for the accepted strict Task-1 outer predictions."""
from __future__ import annotations

from collections import Counter, defaultdict
import json
import re

import numpy as np
import torch
from sklearn.metrics import confusion_matrix

from configs.config import config
from inference.task1_polarity_v63 import polarity_candidate


OUTPUT = config.OUTPUT_DIR / "task1_boundary_errors_v76"
RESULTS = OUTPUT / "results.json"
MARKERS = {
    "attempt_language": re.compile(
        r"(?i)\b(?:attempt\w*|tried|overdos\w*|surviv\w*|woke\s+up|hospitali[sz]\w*)\b"),
    "plan_or_means": re.compile(
        r"(?i)\b(?:plan\w*|going\s+to|gonna|will|pills?|rope|gun|knife|razor|"
        r"jump\w*|hang\w*|method)\b"),
    "negation": re.compile(r"(?i)\b(?:don['’]?t|do\s+not|never|not|no\s+longer)\b"),
    "third_person": re.compile(
        r"(?i)\b(?:he|she|they|friend|brother|sister|mother|father|mom|dad)\b"),
    "hypothetical": re.compile(r"(?i)\b(?:if|would|could|might)\b"),
    "self_harm": re.compile(r"(?i)\b(?:self[- ]?harm\w*|cut\w*|wrist\w*|hurt\s+myself)\b"),
}


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    records = torch.load(
        config.OUTPUT_DIR / "task1_evidence_v4" / "fold0_raw.pt",
        map_location="cpu", weights_only=False)["records"]
    truth = []; prediction = []; details = []; by_pair = defaultdict(Counter)
    for row in records:
        actual = int(row["truth"]); predicted = int(row["risk"])
        polarity = polarity_candidate(row["text"], predicted)
        if polarity is not None:
            predicted = int(polarity[0])
        truth.append(actual); prediction.append(predicted)
        if actual == predicted:
            continue
        markers = [name for name, pattern in MARKERS.items()
                   if pattern.search(str(row["text"]))]
        by_pair[(actual, predicted)].update(markers)
        details.append({"row_id": row["row_id"], "truth": config.ID2RISK[actual],
                        "prediction": config.ID2RISK[predicted],
                        "markers": markers, "gold_evidence": list(row["gold"])})
    payload = {
        "evaluation": "accepted strict outer user holdout after V63 polarity",
        "confusion": confusion_matrix(truth, prediction, labels=np.arange(4)).tolist(),
        "errors": len(details),
        "error_pair_counts": dict(Counter(
            f"{row['truth']}->{row['prediction']}" for row in details)),
        "marker_counts_by_error_pair": {
            f"{config.ID2RISK[a]}->{config.ID2RISK[b]}": dict(values)
            for (a, b), values in by_pair.items()},
        "details": details,
    }
    RESULTS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "details"},
                     indent=2), flush=True)
    return payload


if __name__ == "__main__":
    main()
