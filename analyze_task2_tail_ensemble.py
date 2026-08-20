"""Leak-free ablation of the tail model against the selected Task 2 ensemble."""
import json
import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit
from torch.utils.data import DataLoader, Subset

from baseline import _vectorizer, _fit_factor_models, _factor_probabilities
from configs.config import config
from models.factor_model import MentalRobertaFactorModel
from preprocess.preprocess import load_train_data
from trainer.factor_train import FactorDataset, _collate
from utils.factor_calibration import apply_prior_topk


@torch.no_grad()
def probabilities(checkpoint, dataset, indices, device):
    loader = DataLoader(
        Subset(dataset, indices), batch_size=config.BATCH_SIZE, shuffle=False,
        collate_fn=_collate, num_workers=0,
    )
    model = MentalRobertaFactorModel().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device)); model.eval()
    result = []
    for batch in loader:
        logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        result.append(torch.sigmoid(logits).cpu().numpy())
    del model
    if device.type == "cuda": torch.cuda.empty_cache()
    return np.vstack(result)


def main():
    frame = load_train_data()
    y = np.vstack(frame.factor_vector.to_numpy())
    dataset = FactorDataset(config.CACHE_DIR / "factor_train_cache.pt")
    risk = np.asarray([x["risk_label"] for x in dataset.data])
    groups = np.asarray([x["anon_user_id"] for x in dataset.data])
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=config.SEED)
    outer_train, valid_idx = next(outer.split(np.zeros(len(dataset)), risk, groups))
    inner = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=config.SEED + 91)
    fit_rel, _ = next(inner.split(outer_train, groups=groups[outer_train]))
    fit_idx = outer_train[fit_rel]
    prevalence = y[outer_train].mean(0)

    vectorizer = _vectorizer()
    fit_x = vectorizer.fit_transform(frame.text.iloc[fit_idx])
    valid_x = vectorizer.transform(frame.text.iloc[valid_idx])
    cpu = _factor_probabilities(_fit_factor_models(fit_x, y[fit_idx]), valid_x)
    device = torch.device(config.DEVICE)
    semantic = probabilities(
        config.OUTPUT_DIR / "factor_strict_model_semantic.pt", dataset, valid_idx, device
    )
    legacy = probabilities(
        config.OUTPUT_DIR / "factor_strict_model_asl.pt", dataset, valid_idx, device
    )
    tail_saved = np.load(config.OUTPUT_DIR / "factor_tail_strict_probabilities.npz")
    if not np.array_equal(tail_saved["valid_indices"], valid_idx):
        raise ValueError("Tail probabilities use a different validation fold")
    tail = tail_saved["valid_probability"]
    base = 0.50 * semantic + 0.25 * legacy + 0.25 * cpu
    results = []
    for tail_weight in np.linspace(0.0, 0.50, 11):
        blended = (1.0 - tail_weight) * base + tail_weight * tail
        for ratio in (0.9, 1.0, 1.1, 1.25, 1.4):
            prediction = apply_prior_topk(blended, prevalence, ratio)
            results.append({
                "tail_weight": float(tail_weight), "topk_ratio": ratio,
                "macro_f1": f1_score(y[valid_idx], prediction, average="macro", zero_division=0),
            })
    results.sort(key=lambda x: x["macro_f1"], reverse=True)
    print(json.dumps(results[:15], indent=2))
    (config.OUTPUT_DIR / "factor_tail_ensemble_results.json").write_text(
        json.dumps({"best": results[0], "top15": results[:15]}, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
