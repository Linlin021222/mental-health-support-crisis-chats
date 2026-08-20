"""Create the four-fold-selected Qwen3-Reranker V3 leaderboard candidate.

The accepted Task 1 columns and the established Task 2 base probability are
left intact.  Only Task 2 ranking is blended: 60% current heterogeneous base
rank plus 40% Qwen3-Reranker rank, followed by the unchanged 1.10 prevalence
decoder.  The 0.40 weight was independently selected in every leave-one-fold-
out experiment over the four available user-disjoint folds (1,277 posts).
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd

from configs.config import config


ROOT = Path(__file__).resolve().parent
BASE_SUBMISSION = config.OUTPUT_DIR / "panda_task1_qwen_blend_official_07829.csv"
BASE_TEST = config.OUTPUT_DIR / "factor_signed_graph_stack_v21" / "test_diagnostic.npz"
QWEN_TEST = (
    config.OUTPUT_DIR / "task2_qwen3_reranker_v3" / "full"
    / "qwen3-reranker-8b-factor-v3_test_probabilities.npz"
)
OUTPUT = config.OUTPUT_DIR / "panda_task1_07829_task2_reranker_v3_w040.csv"
MANIFEST = (
    config.OUTPUT_DIR / "task2_qwen3_reranker_v3"
    / "leaderboard_blend_w040_manifest.json"
)

QWEN_WEIGHT = 0.40
PREVALENCE_RATIO = 1.10


def rank_columns(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    ranked = np.empty_like(values)
    denominator = max(1, len(values) - 1)
    for label in range(values.shape[1]):
        order = np.argsort(values[:, label], kind="stable")
        local = np.empty(len(values), dtype=np.float32)
        local[order] = np.arange(len(values), dtype=np.float32) / denominator
        ranked[:, label] = local
    return ranked


def topk(values: np.ndarray, prevalence: float, ratio: float) -> np.ndarray:
    count = max(1, min(
        len(values), int(round(len(values) * float(prevalence) * float(ratio)))
    ))
    chosen = np.argpartition(values, len(values) - count)[len(values) - count:]
    result = np.zeros(len(values), dtype=bool)
    result[chosen] = True
    return result


def parse_factor_predictions(frame: pd.DataFrame) -> np.ndarray:
    result = np.zeros((len(frame), config.NUM_FACTORS), dtype=bool)
    for row, value in enumerate(frame["factors"]):
        labels = ast.literal_eval(str(value))
        for label in labels:
            if label not in config.FACTOR2ID:
                raise ValueError(f"Unknown factor in base submission: {label!r}")
            result[row, config.FACTOR2ID[label]] = True
    return result


def align(rows, saved_rows, probability):
    lookup = {str(row_id): index for index, row_id in enumerate(saved_rows)}
    missing = [str(row_id) for row_id in rows if str(row_id) not in lookup]
    if missing:
        raise RuntimeError(f"Probability file is missing row ids: {missing[:5]}")
    return np.vstack([probability[lookup[str(row_id)]] for row_id in rows])


def main():
    submission = pd.read_csv(BASE_SUBMISSION)
    expected_columns = ["row_id", "risk_level", "evidence", "factors"]
    if submission.columns.tolist() != expected_columns:
        raise RuntimeError(f"Unexpected submission columns: {submission.columns.tolist()}")
    if len(submission) != 378 or submission.row_id.nunique() != len(submission):
        raise RuntimeError("Expected exactly 378 unique leaderboard rows")

    base_saved = np.load(BASE_TEST, allow_pickle=True)
    base_probability = align(
        submission.row_id.astype(str),
        base_saved["row_ids"],
        base_saved["v3"].astype(np.float32),
    )
    qwen_saved = np.load(QWEN_TEST, allow_pickle=False)
    qwen_factors = qwen_saved["factors"].astype(str).tolist()
    if qwen_factors != list(config.FACTOR_LABELS):
        raise RuntimeError("Qwen factor order does not match competition order")
    qwen_probability = align(
        submission.row_id.astype(str),
        qwen_saved["row_id"],
        qwen_saved["probabilities"].astype(np.float32),
    )
    if not np.isfinite(qwen_probability).all():
        raise RuntimeError("Qwen probabilities contain NaN or infinity")

    training = np.load(
        ROOT / "kaggle_factor_reranker_package" / "factor_baseline_oof.npz",
        allow_pickle=True,
    )
    prevalence = training["targets"].astype(np.float32).mean(0)
    base_rank = rank_columns(base_probability)
    qwen_rank = rank_columns(qwen_probability)
    blended_rank = (
        (1.0 - QWEN_WEIGHT) * base_rank + QWEN_WEIGHT * qwen_rank
    )

    baseline_prediction = np.column_stack([
        topk(base_rank[:, label], prevalence[label], PREVALENCE_RATIO)
        for label in range(config.NUM_FACTORS)
    ])
    candidate_prediction = np.column_stack([
        topk(blended_rank[:, label], prevalence[label], PREVALENCE_RATIO)
        for label in range(config.NUM_FACTORS)
    ])

    # This assertion proves that the continuous base and decoder reproduce the
    # exact official Task 2 factor column before Qwen is introduced.
    submitted_baseline = parse_factor_predictions(submission)
    if not np.array_equal(baseline_prediction, submitted_baseline):
        mismatches = int((baseline_prediction != submitted_baseline).sum())
        raise RuntimeError(
            f"Base probability/decoder does not reproduce official factors: "
            f"{mismatches} mismatched label assignments"
        )

    result = submission.copy()
    result["factors"] = [
        str([
            config.ID2FACTOR[label]
            for label, present in enumerate(row) if present
        ])
        for row in candidate_prediction
    ]
    # Task 1 must be byte-equivalent at the dataframe-value level.
    if not result[["row_id", "risk_level", "evidence"]].equals(
        submission[["row_id", "risk_level", "evidence"]]
    ):
        raise RuntimeError("Task 1 columns changed during Task 2 blending")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT, index=False)
    changed = candidate_prediction != baseline_prediction
    manifest = {
        "training_version": "qwen3-reranker-risk-protective-v3-fourfold-blend",
        "evaluation": (
            "four available user-disjoint folds; global weight selected on the "
            "other three folds and evaluated on the held-out fourth"
        ),
        "covered_oof_posts": 1277,
        "missing_fold": 4,
        "crossfit_selected_weights": [0.40, 0.40, 0.40, 0.40],
        "oof_baseline_macro_f1": 0.603807572214538,
        "oof_candidate_macro_f1": 0.6247587780627609,
        "oof_delta": 0.020951205848222965,
        "base_weight": 1.0 - QWEN_WEIGHT,
        "qwen_weight": QWEN_WEIGHT,
        "prevalence_ratio": PREVALENCE_RATIO,
        "task1_source": str(BASE_SUBMISSION),
        "base_probability_source": str(BASE_TEST),
        "qwen_probability_source": str(QWEN_TEST),
        "output": str(OUTPUT),
        "changed_rows": int(changed.any(1).sum()),
        "changed_label_assignments": int(changed.sum()),
        "changes_by_label": {
            config.ID2FACTOR[label]: int(changed[:, label].sum())
            for label in range(config.NUM_FACTORS)
        },
        "baseline_mean_factors": float(baseline_prediction.sum(1).mean()),
        "candidate_mean_factors": float(candidate_prediction.sum(1).mean()),
        "baseline_empty_posts": int((baseline_prediction.sum(1) == 0).sum()),
        "candidate_empty_posts": int((candidate_prediction.sum(1) == 0).sum()),
        "per_label_positive_counts_preserved": bool(np.array_equal(
            baseline_prediction.sum(0), candidate_prediction.sum(0)
        )),
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print(f"Saved submission: {OUTPUT}")


if __name__ == "__main__":
    main()

