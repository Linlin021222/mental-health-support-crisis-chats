"""Higher-precision (0.30) sentence-evidence threshold ablation."""
from configs.config import config
import trainer.factor_sentence_evidence_v16 as experiment


def train_fold0():
    output = config.OUTPUT_DIR / "factor_sentence_evidence_v18"
    experiment.OUTPUT = output
    experiment.RESULTS = output / "fold0_results.json"
    experiment.PSEUDO_EVIDENCE = output / "fold0_pseudo_evidence.jsonl"
    experiment.CHECKPOINT = output / "fold0_model.pt"
    experiment.VALID_PREDICTIONS = output / "fold0_valid.npz"
    experiment.TRAINING_VERSION = "high-precision-sentence-factor-evidence-v18"
    experiment.MIN_EVIDENCE_SCORE = 0.30
    return experiment.train_fold0()


if __name__ == "__main__":
    train_fold0()
