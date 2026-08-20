"""Confidence-filtered continuation of the V16 sentence evidence experiment."""
from configs.config import config
import trainer.factor_sentence_evidence_v16 as experiment


def train_fold0():
    output = config.OUTPUT_DIR / "factor_sentence_evidence_v17"
    experiment.OUTPUT = output
    experiment.RESULTS = output / "fold0_results.json"
    experiment.PSEUDO_EVIDENCE = output / "fold0_pseudo_evidence.jsonl"
    experiment.CHECKPOINT = output / "fold0_model.pt"
    experiment.VALID_PREDICTIONS = output / "fold0_valid.npz"
    experiment.TRAINING_VERSION = "confidence-filtered-sentence-factor-evidence-v17"
    # Manual audit found that forced low-score selections included statements
    # such as "im fine".  0.20 removes those obvious mismatches while retaining
    # 85% of the mined sentences and all original post-level supervision in V3.
    experiment.MIN_EVIDENCE_SCORE = 0.20
    return experiment.train_fold0()


if __name__ == "__main__":
    train_fold0()
