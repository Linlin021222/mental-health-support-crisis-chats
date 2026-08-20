"""Broader long-tail split for the integrated V38 architecture."""
from __future__ import annotations

from configs.config import config
from trainer import factor_definition_retrieval_v38 as experiment


def main():
    output = config.OUTPUT_DIR / "factor_definition_retrieval_tail_v39"
    output.mkdir(parents=True, exist_ok=True)
    # Reuse task-specific base features; only the LSFA/HTTN tail assignment
    # changes in this controlled ablation.
    experiment.OUTPUT = output
    experiment.RESULTS = output / "fold0_results.json"
    experiment.CHECKPOINT = output / "fold0_model.pt"
    experiment.PREDICTIONS = output / "fold0_valid.npz"
    experiment.FEATURES = config.OUTPUT_DIR / "factor_definition_retrieval_v38" / "fold0_base_features.npz"
    experiment.TAIL_THRESHOLD = 100
    return experiment.main()


if __name__ == "__main__":
    main()
