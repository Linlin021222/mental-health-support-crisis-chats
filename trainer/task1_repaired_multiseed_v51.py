"""Strict gate for multi-seed evidence models trained on repaired alignments."""
from __future__ import annotations

from configs.config import config
import trainer.task1_seed_evidence_v28 as experiment


OUTPUT = config.OUTPUT_DIR / "task1_repaired_multiseed_v51"


def main():
    """Reuse the pre-registered V28 recipe with fresh alignment-aware seeds."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    experiment.OUTPUT = OUTPUT
    experiment.RESULTS = OUTPUT / "results.json"
    experiment.CALIBRATION = OUTPUT / "calibration.json"
    experiment.TRAINING_VERSION = "task1-repaired-evidence-multiseed-v51"
    experiment.EXTRA_SEEDS = (424242, 515151)
    return experiment.train_task1_seed_evidence_v28()


if __name__ == "__main__":
    main()
