"""Leak-free sparse/contextual ensemble ablation on the outer user holdout."""
import numpy as np
import torch
from sklearn.metrics import f1_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit
from sklearn.multioutput import ClassifierChain
from torch.utils.data import DataLoader, Subset

from configs.config import config
from preprocess.preprocess import load_train_data
from baseline import _vectorizer, _fit_factor_models, _factor_probabilities
from trainer.factor_train import FactorDataset, _collate, STRICT_CHECKPOINT
from models.factor_model import MentalRobertaFactorModel
from datasets.dataset import SuicideRiskDataset
from datasets.collator import SuicideRiskCollator
from models.multitask_model import SuicideRiskMultiTaskModel
from utils.factor_calibration import calibrate_factor_thresholds, apply_prior_topk


@torch.no_grad()
def mental_probs(model, dataset, indices, device):
    loader = DataLoader(Subset(dataset, indices), batch_size=config.BATCH_SIZE, shuffle=False,
                        collate_fn=_collate, num_workers=0)
    values = []
    for batch in loader:
        logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        values.append(torch.sigmoid(logits).cpu().numpy())
    return np.vstack(values)


@torch.no_grad()
def multitask_probs(model, dataset, indices, device):
    loader = DataLoader(
        Subset(dataset, indices), batch_size=config.BATCH_SIZE, shuffle=False,
        collate_fn=SuicideRiskCollator(), num_workers=0,
    )
    values = []
    for batch in loader:
        output = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        values.append(torch.sigmoid(output["factor_logits"]).cpu().numpy())
    return np.vstack(values)


def score_calibrated(cal_y, cal_p, valid_y, valid_p, prevalence):
    thresholds = calibrate_factor_thresholds(cal_y, cal_p, reference_prevalence=prevalence)
    return f1_score(valid_y, valid_p >= thresholds[None, :], average="macro", zero_division=0)


def main():
    dataset = FactorDataset(config.CACHE_DIR / "factor_train_cache.pt")
    frame = load_train_data()
    risk = np.asarray([x["risk_label"] for x in dataset.data])
    groups = np.asarray([x["anon_user_id"] for x in dataset.data])
    outer = StratifiedGroupKFold(n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED)
    outer_train, valid_idx = next(outer.split(np.zeros(len(dataset)), risk, groups))
    inner = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=config.SEED + 91)
    fit_rel, cal_rel = next(inner.split(outer_train, groups=groups[outer_train]))
    fit_idx, cal_idx = outer_train[fit_rel], outer_train[cal_rel]
    y = np.vstack(frame.factor_vector.to_numpy())
    prevalence = y[fit_idx].mean(0)

    vectorizer = _vectorizer()
    fit_x = vectorizer.fit_transform(frame.text.iloc[fit_idx])
    cal_x = vectorizer.transform(frame.text.iloc[cal_idx])
    valid_x = vectorizer.transform(frame.text.iloc[valid_idx])
    cpu_models = _fit_factor_models(fit_x, y[fit_idx])
    cpu_cal = _factor_probabilities(cpu_models, cal_x)
    cpu_valid = _factor_probabilities(cpu_models, valid_x)
    chain_base = LogisticRegression(
        C=2.0, class_weight="balanced", max_iter=500, solver="liblinear"
    )
    chain_predictions = []
    for order in (None, "random"):
        chain = ClassifierChain(
            chain_base, order=order, chain_method="predict_proba",
            random_state=config.SEED,
        )
        chain.fit(fit_x, y[fit_idx])
        chain_predictions.append(chain.predict_proba(valid_x))
    chain_valid = np.mean(chain_predictions, axis=0)

    device = torch.device(config.DEVICE)
    model = MentalRobertaFactorModel().to(device)
    model.load_state_dict(torch.load(STRICT_CHECKPOINT, map_location=device)); model.eval()
    mental_cal = mental_probs(model, dataset, cal_idx, device)
    mental_valid = mental_probs(model, dataset, valid_idx, device)

    old_model = MentalRobertaFactorModel().to(device)
    old_model.load_state_dict(torch.load(
        config.OUTPUT_DIR / "factor_strict_model_asl.pt", map_location=device
    )); old_model.eval()
    old_cal = mental_probs(old_model, dataset, cal_idx, device)
    old_valid = mental_probs(old_model, dataset, valid_idx, device)
    del model, old_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    multitask_dataset = SuicideRiskDataset(config.CACHE_DIR / "train_cache.pt")
    multitask_model = SuicideRiskMultiTaskModel().to(device)
    multitask_model.load_state_dict(torch.load(
        config.OUTPUT_DIR / "best_model.pt", map_location=device
    ))
    multitask_model.eval()
    multitask_valid = multitask_probs(multitask_model, multitask_dataset, valid_idx, device)
    del multitask_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    results = {}
    for alpha in np.linspace(0, 1, 11):
        cal_p = alpha * mental_cal + (1 - alpha) * cpu_cal
        valid_p = alpha * mental_valid + (1 - alpha) * cpu_valid
        results[f"blend_mental_{alpha:.1f}"] = score_calibrated(
            y[cal_idx], cal_p, y[valid_idx], valid_p, prevalence
        )
        for ratio in (0.9, 1.0, 1.1, 1.25):
            pred = apply_prior_topk(valid_p, y[outer_train].mean(0), ratio)
            results[f"blend_{alpha:.1f}_topk_{ratio:.2f}"] = f1_score(
                y[valid_idx], pred, average="macro", zero_division=0
            )
    # Three-way ensemble: semantic-alignment MentalRoBERTa, original ASL
    # MentalRoBERTa, and sparse TF-IDF. Coarse weights reduce calibration
    # overfitting while testing whether the two neural objectives complement.
    for new_weight in (0.0, 0.25, 0.5, 0.75, 1.0):
        for old_weight in (0.0, 0.25, 0.5, 0.75, 1.0):
            cpu_weight = 1.0 - new_weight - old_weight
            if cpu_weight < 0:
                continue
            valid_p = (new_weight * mental_valid + old_weight * old_valid
                       + cpu_weight * cpu_valid)
            for ratio in (0.9, 1.0, 1.1, 1.25, 1.4):
                pred = apply_prior_topk(valid_p, y[outer_train].mean(0), ratio)
                results[f"three_new{new_weight:.2f}_old{old_weight:.2f}_cpu{cpu_weight:.2f}_r{ratio:.2f}"] = f1_score(
                    y[valid_idx], pred, average="macro", zero_division=0
                )
    best_three_valid = 0.50 * mental_valid + 0.25 * old_valid + 0.25 * cpu_valid
    for chain_weight in np.linspace(0.0, 1.0, 11):
        valid_p = chain_weight * chain_valid + (1.0 - chain_weight) * best_three_valid
        for ratio in (0.9, 1.0, 1.1, 1.25, 1.4):
            pred = apply_prior_topk(valid_p, y[outer_train].mean(0), ratio)
            results[f"chain{chain_weight:.1f}_best3{1-chain_weight:.1f}_r{ratio:.2f}"] = f1_score(
                y[valid_idx], pred, average="macro", zero_division=0
            )
    # The DeBERTa multi-task factor head has received auxiliary risk and
    # evidence supervision. Test it as a fourth, architecturally distinct
    # signal with a deliberately coarse simplex search.
    grid = (0.0, 0.25, 0.5, 0.75, 1.0)
    for new_weight in grid:
        for old_weight in grid:
            for cpu_weight in grid:
                multitask_weight = 1.0 - new_weight - old_weight - cpu_weight
                if multitask_weight < -1e-8:
                    continue
                valid_p = (
                    new_weight * mental_valid + old_weight * old_valid
                    + cpu_weight * cpu_valid + multitask_weight * multitask_valid
                )
                for ratio in (0.9, 1.0, 1.1, 1.25, 1.4):
                    pred = apply_prior_topk(valid_p, y[outer_train].mean(0), ratio)
                    name = (
                        f"four_new{new_weight:.2f}_old{old_weight:.2f}_"
                        f"cpu{cpu_weight:.2f}_multi{multitask_weight:.2f}_r{ratio:.2f}"
                    )
                    results[name] = f1_score(
                        y[valid_idx], pred, average="macro", zero_division=0
                    )
    # Select one sparse/contextual mixing weight per factor using only the
    # inner calibration users. Average precision avoids coupling selection to
    # a particular threshold.
    selected_alpha = np.full(config.NUM_FACTORS, 0.5, dtype=np.float32)
    adaptive_valid = np.zeros_like(mental_valid)
    for j in range(config.NUM_FACTORS):
        if y[cal_idx, j].sum() >= 3:
            candidates = []
            for alpha in np.linspace(0, 1, 11):
                blended = alpha * mental_cal[:, j] + (1 - alpha) * cpu_cal[:, j]
                candidates.append((average_precision_score(y[cal_idx, j], blended), float(alpha)))
            selected_alpha[j] = max(candidates, key=lambda pair: pair[0])[1]
        adaptive_valid[:, j] = (
            selected_alpha[j] * mental_valid[:, j]
            + (1 - selected_alpha[j]) * cpu_valid[:, j]
        )
    for ratio in (0.9, 1.0, 1.1, 1.25, 1.4, 1.6):
        pred = apply_prior_topk(adaptive_valid, y[outer_train].mean(0), ratio)
        results[f"per_label_ap_topk_{ratio:.2f}"] = f1_score(
            y[valid_idx], pred, average="macro", zero_division=0
        )
    print("selected MentalRoBERTa weights:", {
        config.ID2FACTOR[j]: float(selected_alpha[j]) for j in range(config.NUM_FACTORS)
    })
    for name, value in sorted(results.items(), key=lambda x: x[1], reverse=True)[:20]:
        print(f"{name}: {value:.6f}")
    best = max(results, key=results.get)
    print(f"BEST={best} SCORE={results[best]:.6f}")


if __name__ == "__main__":
    main()
