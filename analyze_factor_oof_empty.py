"""OOF ablation for prevalence top-k plus controlled empty-row filling."""
import json
import numpy as np
from sklearn.metrics import f1_score

from configs.config import config
from utils.factor_calibration import apply_prior_topk


def fill_empty(probability, prediction, allowed_empty):
    result = prediction.copy()
    empty = np.flatnonzero(result.sum(1) == 0)
    fill_count = max(0, len(empty) - allowed_empty)
    if not fill_count:
        return result
    # Compare labels on percentile ranks, since neural/logistic probability
    # scales differ strongly between factor labels.
    ranks = np.empty_like(probability, dtype=np.float32)
    for j in range(probability.shape[1]):
        order = np.argsort(probability[:, j])
        ranks[order, j] = np.linspace(0.0, 1.0, len(probability), dtype=np.float32)
    confidence = ranks[empty].max(1)
    chosen_rows = empty[np.argsort(confidence)[-fill_count:]]
    chosen_labels = ranks[chosen_rows].argmax(1)
    result[chosen_rows, chosen_labels] = True
    return result


def main():
    saved = np.load(config.OUTPUT_DIR / "factor_cv" / "factor_oof_predictions.npz")
    targets = saved["targets"]
    prevalence = targets.mean(0)
    gold_empty = int((targets.sum(1) == 0).sum())
    results = []
    for semantic_weight in (0.5, 0.6, 0.7, 0.8):
        probability = semantic_weight*saved["semantic"] + (1-semantic_weight)*saved["cpu"]
        for ratio in (0.9, 1.0, 1.1, 1.25, 1.4):
            raw = apply_prior_topk(probability, prevalence, ratio)
            for empty_multiplier in (None, 1.0, 1.5, 2.0):
                prediction = raw if empty_multiplier is None else fill_empty(
                    probability, raw, int(round(gold_empty*empty_multiplier))
                )
                results.append({
                    "semantic_weight": semantic_weight, "ratio": ratio,
                    "empty_multiplier": empty_multiplier,
                    "empty_rows": int((prediction.sum(1) == 0).sum()),
                    "macro_f1": float(f1_score(
                        targets, prediction, average="macro", zero_division=0
                    )),
                })
    results.sort(key=lambda x: x["macro_f1"], reverse=True)
    print(json.dumps(results[:20], indent=2))
    (config.OUTPUT_DIR / "factor_cv" / "empty_fill_results.json").write_text(
        json.dumps({"gold_empty": gold_empty, "best": results[0], "top20": results[:20]}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
