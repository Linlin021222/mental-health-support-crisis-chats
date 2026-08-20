"""Import explicitly reviewed rows from the bilingual V19 workbook."""
from __future__ import annotations

import json

import pandas as pd

from configs.config import config


OUTPUT = config.OUTPUT_DIR / "factor_boundary_review_v19"
WORKBOOK = OUTPUT / "factor_boundary_review_bilingual.xlsx"
SOURCE = OUTPUT / "review_records.json"
CONFIRMED = OUTPUT / "human_confirmed_reviews.json"
IMPORT_SUMMARY = OUTPUT / "human_review_import_summary.json"
DECISIONS = {"接受": "accept", "拒绝": "reject", "不确定": "uncertain"}


def import_reviews():
    if not WORKBOOK.exists():
        raise FileNotFoundError(f"Missing review workbook: {WORKBOOK}")
    sheet = pd.read_excel(WORKBOOK, sheet_name="边界句审核")
    required = {"Review ID", "最终结论（可编辑）", "审核备注", "已人工查看"}
    missing = required - set(sheet.columns)
    if missing:
        raise ValueError(f"Review workbook is missing columns: {sorted(missing)}")
    source = {int(row["review_id"]): row for row in
              json.loads(SOURCE.read_text(encoding="utf-8"))}
    confirmed = []
    for item in sheet.to_dict("records"):
        identifier = int(item["Review ID"])
        if identifier not in source:
            raise ValueError(f"Unknown Review ID: {identifier}")
        reviewed = str(item["已人工查看"]).strip() == "是"
        if not reviewed:
            continue
        value = str(item["最终结论（可编辑）"]).strip()
        if value not in DECISIONS:
            raise ValueError(f"Invalid final decision for Review ID {identifier}: {value}")
        row = dict(source[identifier])
        row["human_final"] = DECISIONS[value]
        note = item.get("审核备注", "")
        row["human_note"] = "" if pd.isna(note) else str(note).strip()
        row["explicitly_reviewed"] = True
        confirmed.append(row)
    payload = {
        "confirmed_records": len(confirmed),
        "decisions": {decision: sum(row["human_final"] == decision for row in confirmed)
                      for decision in ("accept", "reject", "uncertain")},
        "source_workbook": str(WORKBOOK),
        "safe_for_training": bool(confirmed),
    }
    CONFIRMED.write_text(json.dumps(confirmed, ensure_ascii=False, indent=2), encoding="utf-8")
    IMPORT_SUMMARY.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    return payload


if __name__ == "__main__":
    import_reviews()
