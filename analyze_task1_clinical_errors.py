"""Export strict Task 1 risk/evidence errors for clinical-pattern analysis."""
from __future__ import annotations

import csv
import sys

import torch

from baseline import _post_phrase_f1
from configs.config import config
from inference.task1_evidence_v4 import (
    apply_evidence_policy, decode_model_evidence, load_evidence_calibration,
)


RAW = config.OUTPUT_DIR / "task1_evidence_v4" / "fold0_raw.pt"
OUTPUT = config.OUTPUT_DIR / "task1_clinical" / "strict_errors.csv"


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    raw = torch.load(RAW, map_location="cpu", weights_only=False)
    records = raw["records"]
    calibration = load_evidence_calibration()
    if calibration is None:
        raise FileNotFoundError("Task 1 evidence-v4 calibration is missing or not adopted")
    rows = []
    for record in records:
        model_phrases = decode_model_evidence(
            record["text"], record["offsets"], record["start"], record["end"],
            threshold=float(calibration["threshold"]),
            max_tokens=int(calibration["max_tokens"]),
            end_policy=calibration["end_policy"], limit=5,
        )
        evidence = apply_evidence_policy(
            record["text"], int(record["risk"]), model_phrases,
            policy=calibration["cue_policy"], topk=int(calibration["topk"]),
        )
        rows.append({
            "row_id": record["row_id"],
            "anon_user_id": record["user"],
            "gold_risk": config.ID2RISK[int(record["truth"])],
            "pred_risk": config.ID2RISK[int(record["risk"])],
            "risk_error": int(record["truth"] != record["risk"]),
            "gold_evidence": "; ".join(record["gold"]),
            "model_evidence": "; ".join(model_phrases),
            "pred_evidence": "; ".join(evidence),
            "phrase_f1": _post_phrase_f1(evidence, record["gold"]),
            "post": record["text"],
        })
    rows.sort(key=lambda row: (
        -row["risk_error"], row["gold_risk"], row["pred_risk"], row["row_id"]
    ))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    errors = [row for row in rows if row["risk_error"]]
    for row in errors:
        print("=" * 100)
        print(f"row={row['row_id']} {row['gold_risk']} -> {row['pred_risk']}")
        print(f"gold evidence: {row['gold_evidence']}")
        print(row["post"].replace("\n", " ")[:1200])
    print(f"\nSaved {len(errors)} risk errors of {len(rows)} records to {OUTPUT}")
    return OUTPUT


if __name__ == "__main__":
    main()
