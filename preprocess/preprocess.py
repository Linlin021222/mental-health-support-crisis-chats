"""Read competition workbooks without changing the text used for evidence offsets."""
import ast
import re
import pandas as pd
import numpy as np
from configs.config import config


def text_value(value):
    return "" if pd.isna(value) else str(value)


def parse_risk_label(value):
    """Normalize case and invisible/newline whitespace in Excel labels."""
    key = re.sub(r"\s+", "", text_value(value)).casefold()
    mapping = {name.casefold(): code for name, code in config.RISK_LABELS.items()}
    return mapping.get(key)


def parse_evidence(value):
    if pd.isna(value):
        return []
    # The published data uses semicolon-separated evidence.  Do not split on
    # commas: a valid evidence phrase can itself contain a comma.
    parts = [part.strip() for part in str(value).split(";") if part.strip()]
    # The workbook uses textual sentinels for posts with no suicide evidence;
    # they are not literal spans in the Reddit post.
    empty_markers = {"none", "nan", "null", "n/a", "no evidence", "[]"}
    return [part for part in parts if part.casefold() not in empty_markers]


def parse_factors(value):
    return (parse_factor_counts(value) > 0).astype(np.float32)


def parse_factor_counts(value):
    """Preserve repeated factor annotations as an auxiliary salience signal."""
    if pd.isna(value):
        return np.zeros(config.NUM_FACTORS, dtype=np.float32)
    raw = value
    if isinstance(raw, str):
        try:
            raw = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            raw = [part.strip() for part in re.split(r"[;,]", raw)]
    if not isinstance(raw, (list, tuple, set)):
        raw = [raw]
    result = np.zeros(config.NUM_FACTORS, dtype=np.float32)
    for label in raw:
        key = str(label).strip()
        if key in config.FACTOR2ID:
            result[config.FACTOR2ID[key]] += 1.0
    return result


def load_train_data():
    frame = pd.read_excel(config.TRAIN_FILE)
    required = {"row_id", "post", "suicide risk", "evidence for suicide risk level", "factors"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"train.xlsx is missing columns: {sorted(missing)}")
    result = pd.DataFrame({
        "row_id": frame["row_id"],
        "anon_user_id": frame["anon_user_id"] if "anon_user_id" in frame.columns else frame["row_id"],
        "post_id": frame["post_id"] if "post_id" in frame.columns else frame["row_id"],
        "text": frame["post"].map(text_value),
        "risk_label": frame["suicide risk"].map(parse_risk_label),
        "evidence": frame["evidence for suicide risk level"].map(parse_evidence),
        "factor_vector": frame["factors"].map(parse_factors),
        "factor_counts": frame["factors"].map(parse_factor_counts),
    })
    if result.risk_label.isna().any():
        bad_rows = frame.loc[result.risk_label.isna(), ["row_id", "suicide risk"]].head(10).to_dict("records")
        raise ValueError(f"Unknown suicide-risk labels in train.xlsx: {bad_rows}")
    result["risk_label"] = result["risk_label"].astype(int)
    return result


def load_test_data():
    frame = pd.read_excel(config.TEST_FILE)
    if not {"row_id", "post"}.issubset(frame.columns):
        raise ValueError("leaderboard.xlsx must contain row_id and post")
    return pd.DataFrame({
        "row_id": frame["row_id"],
        "anon_user_id": frame["anon_user_id"] if "anon_user_id" in frame.columns else frame["row_id"],
        "post_id": frame["post_id"] if "post_id" in frame.columns else frame["row_id"],
        "text": frame["post"].map(text_value),
    })
