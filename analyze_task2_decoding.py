"""Evaluate decoding alternatives on the untouched outer user holdout."""
import json
import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader

from configs.config import config
from trainer.factor_train import FactorDataset, _collate, STRICT_CHECKPOINT
from models.factor_model import MentalRobertaFactorModel


def predict_with_prior_caps(probabilities, thresholds, prevalence, floor_ratio=0.0, ceiling_ratio=None):
    pred = probabilities >= thresholds[None, :]
    n = len(pred)
    for j in range(config.NUM_FACTORS):
        order = np.argsort(probabilities[:, j])[::-1]
        minimum = int(round(n * prevalence[j] * floor_ratio))
        if minimum > 0 and pred[:, j].sum() < minimum:
            pred[order[:minimum], j] = True
        if ceiling_ratio is not None:
            maximum = max(1, int(round(n * prevalence[j] * ceiling_ratio)))
            if pred[:, j].sum() > maximum:
                pred[:, j] = False
                pred[order[:maximum], j] = True
    return pred


def topk_prior(probabilities, prevalence, ratio):
    pred = np.zeros_like(probabilities, dtype=bool)
    n = len(pred)
    for j in range(config.NUM_FACTORS):
        count = max(1, int(round(n * prevalence[j] * ratio)))
        pred[np.argsort(probabilities[:, j])[-count:], j] = True
    return pred


def main():
    cache = config.CACHE_DIR / "factor_train_cache.pt"
    dataset = FactorDataset(cache)
    risk = np.asarray([x["risk_label"] for x in dataset.data])
    groups = np.asarray([x["anon_user_id"] for x in dataset.data])
    outer = StratifiedGroupKFold(n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED)
    train_idx, valid_idx = next(outer.split(np.zeros(len(dataset)), risk, groups))
    loader = DataLoader(torch.utils.data.Subset(dataset, valid_idx), batch_size=config.BATCH_SIZE,
                        shuffle=False, collate_fn=_collate, num_workers=0)
    device = torch.device(config.DEVICE)
    model = MentalRobertaFactorModel().to(device)
    model.load_state_dict(torch.load(STRICT_CHECKPOINT, map_location=device))
    model.eval(); probs = []; truth = []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            probs.append(torch.sigmoid(logits).cpu().numpy())
            truth.append(batch["factor_vectors"].numpy())
    probs, truth = np.vstack(probs), np.vstack(truth)
    train_targets = np.vstack([dataset.data[int(i)]["factor_vector"].numpy() for i in train_idx])
    prevalence = train_targets.mean(0)
    payload = json.loads((config.OUTPUT_DIR / "factor_strict_calibration.json").read_text(encoding="utf-8"))
    thresholds = np.asarray([payload["thresholds"][x] for x in config.FACTOR_LABELS])
    results = {}
    base = probs >= thresholds[None, :]
    results["calibrated_thresholds"] = f1_score(truth, base, average="macro", zero_division=0)
    for ratio in (0.75, 0.90, 1.00, 1.10, 1.25):
        pred = topk_prior(probs, prevalence, ratio)
        results[f"prior_topk_{ratio:.2f}"] = f1_score(truth, pred, average="macro", zero_division=0)
    for ceiling in (1.00, 1.25, 1.50, 2.00):
        pred = predict_with_prior_caps(probs, thresholds, prevalence, floor_ratio=0.70,
                                       ceiling_ratio=ceiling)
        results[f"threshold_floor0.70_cap{ceiling:.2f}"] = f1_score(
            truth, pred, average="macro", zero_division=0
        )
    best_name = max(results, key=results.get)
    print(json.dumps(results, indent=2))
    print(f"BEST={best_name} SCORE={results[best_name]:.6f} "
          f"GAIN={results[best_name] - results['calibrated_thresholds']:+.6f}")


if __name__ == "__main__":
    main()
