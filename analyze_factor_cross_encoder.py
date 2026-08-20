"""Report the adopted cross-encoder ensemble on strict five-fold OOF data."""
import json
import numpy as np
from sklearn.metrics import f1_score

from configs.config import config
from inference.factor_nli import _rank_decode
from trainer.factor_cv import OOF_FILE
from trainer.factor_cross_encoder_cv import CROSS_OOF_FILE


OUTPUT = config.OUTPUT_DIR / "factor_cross_encoder" / "per_label_adopted.json"


def analyze():
    old = np.load(OOF_FILE)
    cross = np.load(CROSS_OOF_FILE)
    targets = old["targets"].astype(np.int8)
    base = (config.FACTOR_SEMANTIC_MODEL_WEIGHT * old["semantic"]
            + config.FACTOR_CPU_ENSEMBLE_WEIGHT * old["cpu"])
    mixed = (config.FACTOR_CROSS_BASE_WEIGHT * base
             + config.FACTOR_CROSS_WEIGHT * cross["probabilities"])
    prevalence = targets.mean(0)
    baseline_prediction = _rank_decode(base, prevalence, config.FACTOR_TOPK_RATIO)
    prediction = _rank_decode(mixed, prevalence, config.FACTOR_CROSS_TOPK_RATIO)
    labels = {}
    for j, name in enumerate(config.FACTOR_LABELS):
        before = f1_score(targets[:, j], baseline_prediction[:, j], zero_division=0)
        after = f1_score(targets[:, j], prediction[:, j], zero_division=0)
        labels[name] = {
            "support": int(targets[:, j].sum()), "baseline_f1": float(before),
            "adopted_f1": float(after), "delta": float(after - before),
            "predicted_positive": int(prediction[:, j].sum()),
        }
    payload = {
        "baseline_macro_f1": float(f1_score(
            targets, baseline_prediction, average="macro", zero_division=0
        )),
        "adopted_macro_f1": float(f1_score(
            targets, prediction, average="macro", zero_division=0
        )),
        "mean_labels_per_post": float(prediction.sum(1).mean()),
        "empty_predictions": int((prediction.sum(1) == 0).sum()),
        "per_label": labels,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "per_label"}, indent=2))
    for name, values in labels.items():
        print(f"{name:48s} {values['baseline_f1']:.4f} -> "
              f"{values['adopted_f1']:.4f} ({values['delta']:+.4f})")
    return payload


if __name__ == "__main__":
    analyze()
