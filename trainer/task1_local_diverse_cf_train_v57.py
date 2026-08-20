"""Strict training wrapper for diversity-controlled V57 augmentation."""
from __future__ import annotations

from configs.config import config
from trainer import task1_local_counterfactual_train_v56 as experiment


OUTPUT = config.OUTPUT_DIR / "task1_local_diverse_cf_v57"


def main(force=False):
    experiment.OUTPUT = OUTPUT
    experiment.SYNTHETIC_FILE = OUTPUT / "synthetic.json"
    experiment.RESULTS = OUTPUT / "strict_results.json"
    experiment.CHECKPOINT = OUTPUT / "strict_model.pt"
    experiment.PREDICTIONS = OUTPUT / "strict_predictions.pt"
    experiment.FULL_CHECKPOINT = OUTPUT / "full_model.pt"
    experiment.FULL_MANIFEST = OUTPUT / "full_manifest.json"
    experiment.TRAINING_VERSION = "task1-local-diverse-counterfactual-v57"
    experiment.SEED = 575757
    experiment.SYNTHETIC_REPEAT = 1
    experiment.FIXED_RISK_WEIGHT = .10
    return experiment.main(force=force)


def train_full(force=False):
    # Configure the shared implementation before invoking its full-data path.
    main_configuration = {
        "OUTPUT": OUTPUT,
        "SYNTHETIC_FILE": OUTPUT / "synthetic.json",
        "RESULTS": OUTPUT / "strict_results.json",
        "FULL_CHECKPOINT": OUTPUT / "full_model.pt",
        "FULL_MANIFEST": OUTPUT / "full_manifest.json",
        "TRAINING_VERSION": "task1-local-diverse-counterfactual-v57",
        "SEED": 575757, "SYNTHETIC_REPEAT": 1, "FIXED_RISK_WEIGHT": .10,
    }
    for name, value in main_configuration.items():
        setattr(experiment, name, value)
    return experiment.train_full(force=force)


if __name__ == "__main__":
    main()
