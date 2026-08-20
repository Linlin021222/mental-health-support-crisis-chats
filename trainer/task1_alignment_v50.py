"""Strict fold-0 gate for punctuation/whitespace-aware evidence alignment."""
from __future__ import annotations

from configs.config import config
import trainer.task1_cv as cv


OUTPUT = config.OUTPUT_DIR / "task1_alignment_v50"


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cv.OUTPUT_DIR = OUTPUT
    cv.OOF_FILE = OUTPUT / "oof_predictions.npz"
    cv.RESULT_FILE = OUTPUT / "cv_results.json"
    cv.CALIBRATION_FILE = OUTPUT / "calibration.json"
    cv.TRAINING_VERSION = "task1-rationale-alignment-v50"
    return cv.train_task1_cv(only_fold0=True)


if __name__ == "__main__":
    main()
