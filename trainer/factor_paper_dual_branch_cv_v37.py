"""Five-fold nested validation of the paper-aligned V36 factor branches."""
from __future__ import annotations

import json

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from configs.config import config
from inference.factor_nli import _rank_decode
from preprocess.preprocess import load_train_data
from trainer.factor_llm_lexical_v6 import _current_v3_probability
from trainer.factor_paper_dual_branch_v36 import (
    EPOCHS, PaperDualBranchModel, _freeze_accepted, _predict, _train,
)
from trainer.factor_sentence_evidence_cv_v27 import _user_bootstrap
from trainer.factor_train import FactorDataset
from utils.seed import seed_everything


OUTPUT = config.OUTPUT_DIR / "factor_paper_dual_branch_cv_v37"
RESULTS = OUTPUT / "cv_results.json"
CALIBRATION = OUTPUT / "calibration.json"
OOF = OUTPUT / "oof_predictions.npz"
TRAINING_VERSION = "five-fold-paper-risk-protective-dual-branch-v37"
SOURCE_DIR = config.OUTPUT_DIR / "factor_cv"
WEIGHTS = (0.0, 0.02, 0.05, 0.10)
RATIOS = (1.0, 1.10)


def _rank_columns(probability):
    result = np.zeros_like(probability, dtype=np.float32)
    for label in range(probability.shape[1]):
        order = np.argsort(probability[:, label], kind="mergesort")
        result[order, label] = np.linspace(0., 1., len(probability), dtype=np.float32)
    return result


def _train_fold(fold, train_idx, valid_idx, dataset, targets, device):
    prediction_file = OUTPUT / f"fold{fold}_valid.npz"
    checkpoint = OUTPUT / f"fold{fold}_residual.pt"
    if fold == 0:
        v36 = config.OUTPUT_DIR / "factor_paper_dual_branch_v36" / "fold0_valid.npz"
        if v36.exists() and not prediction_file.exists():
            saved = np.load(v36)
            if np.array_equal(saved["valid_indices"], valid_idx):
                np.savez_compressed(
                    prediction_file, probabilities=saved["probabilities"],
                    valid_indices=valid_idx, history=saved["history"],
                    training_version=TRAINING_VERSION,
                )
    if prediction_file.exists():
        saved = np.load(prediction_file)
        if (str(saved["training_version"]) == TRAINING_VERSION
                and np.array_equal(saved["valid_indices"], valid_idx)):
            print(f"V37 fold {fold}: resumed", flush=True)
            return saved["probabilities"].astype(np.float32), json.loads(str(saved["history"]))
    source = SOURCE_DIR / f"fold{fold}_model.pt"
    if not source.exists():
        raise FileNotFoundError(source)
    seed_everything(config.SEED + 3600 + fold)
    model = PaperDualBranchModel().to(device)
    model.load_state_dict(torch.load(source, map_location="cpu", weights_only=True), strict=False)
    _freeze_accepted(model)
    print(f"V37 fold {fold}: train={len(train_idx)} valid={len(valid_idx)}", flush=True)
    history = _train(model, dataset, train_idx, targets, device)
    probability = _predict(model, dataset, valid_idx, device)
    state = {name: value.detach().cpu() for name, value in model.state_dict().items()
             if name.startswith((
                 "paper_queries", "risk_branch.", "protective_branch.",
                 "risk_residual.", "protective_residual.",
                 "definition_gate", "risk_gate", "protective_gate",
             ))}
    torch.save({"training_version": TRAINING_VERSION, "state_dict": state}, checkpoint)
    np.savez_compressed(
        prediction_file, probabilities=probability, valid_indices=valid_idx,
        history=json.dumps(history), training_version=TRAINING_VERSION,
    )
    del model; torch.cuda.empty_cache()
    return probability, history


def _label_prediction(score, prevalence, ratio):
    count = max(1, min(len(score), int(round(len(score) * prevalence * ratio))))
    prediction = np.zeros(len(score), dtype=bool)
    prediction[np.argpartition(score, len(score) - count)[len(score) - count:]] = True
    return prediction


def _select(current_rank, paper_rank, targets, fit_idx, component_folds):
    weights = np.zeros(config.NUM_FACTORS, dtype=np.float32)
    ratios = np.ones(config.NUM_FACTORS, dtype=np.float32) * 1.10
    audit = []
    for label in range(config.NUM_FACTORS):
        prevalence = targets[fit_idx, label].mean()
        candidates = []
        for weight in WEIGHTS:
            mixed = (1. - weight) * current_rank[:, label] + weight * paper_rank[:, label]
            for ratio in RATIOS:
                pooled = f1_score(
                    targets[fit_idx, label],
                    _label_prediction(mixed[fit_idx], prevalence, ratio), zero_division=0,
                )
                block = []
                for indices in component_folds:
                    block_fit = np.setdiff1d(fit_idx, indices)
                    block_prevalence = targets[block_fit, label].mean()
                    block.append(f1_score(
                        targets[indices, label],
                        _label_prediction(mixed[indices], block_prevalence, ratio),
                        zero_division=0,
                    ))
                candidates.append((pooled, float(np.mean(block)), float(np.min(block)),
                                   -weight, -abs(ratio - 1.10), weight, ratio, block))
        best = max(candidates)
        # Stability gate: the chosen paper weight must not make more than one
        # component fold worse than the same-ratio no-paper baseline.
        weight, ratio, block = best[5], best[6], np.asarray(best[7])
        baseline_blocks = []
        for indices in component_folds:
            block_fit = np.setdiff1d(fit_idx, indices)
            block_prevalence = targets[block_fit, label].mean()
            baseline_blocks.append(f1_score(
                targets[indices, label],
                _label_prediction(current_rank[indices, label], block_prevalence, ratio),
                zero_division=0,
            ))
        worse = int((block < np.asarray(baseline_blocks) - 1e-12).sum())
        improved = int((block > np.asarray(baseline_blocks) + 1e-12).sum())
        if weight > 0 and (worse > 1 or improved < 2):
            weight = 0.0
        weights[label] = weight; ratios[label] = ratio
        audit.append({
            "label": config.ID2FACTOR[label], "weight": float(weight),
            "ratio": float(ratio), "improved_component_folds": improved,
            "worse_component_folds": worse,
        })
    return weights, ratios, audit


def cross_validate():
    if not torch.cuda.is_available():
        raise RuntimeError("V37 requires CUDA")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame = load_train_data().reset_index(drop=True)
    targets = np.vstack(frame.factor_vector.to_numpy()).astype(np.int8)
    risk = frame.risk_label.to_numpy(dtype=np.int64)
    groups = frame.anon_user_id.astype(str).to_numpy()
    folds = list(StratifiedGroupKFold(
        n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED
    ).split(np.zeros(len(frame)), risk, groups))
    dataset = FactorDataset(config.CACHE_DIR / "factor_train_cache.pt")
    device = torch.device("cuda"); paper_oof = np.zeros_like(targets, dtype=np.float32)
    histories = []
    for fold, (train_idx, valid_idx) in enumerate(folds):
        probability, history = _train_fold(
            fold, train_idx, valid_idx, dataset, targets, device
        )
        paper_oof[valid_idx] = probability; histories.append({"fold": fold, "history": history})

    current, _ = _current_v3_probability()
    current_rank, paper_rank = _rank_columns(current), _rank_columns(paper_oof)
    baseline_prediction = np.zeros_like(targets, dtype=bool)
    candidate_prediction = np.zeros_like(targets, dtype=bool)
    fold_rows, fold_weights, fold_ratios = [], [], []
    for outer, (fit_idx, valid_idx) in enumerate(folds):
        components = [folds[i][1] for i in range(config.N_FOLDS) if i != outer]
        weights, ratios, audit = _select(
            current_rank, paper_rank, targets, fit_idx, components
        )
        mixed = ((1. - weights) * current_rank[valid_idx]
                 + weights * paper_rank[valid_idx])
        prevalence = targets[fit_idx].mean(0)
        baseline_prediction[valid_idx] = _rank_decode(
            current_rank[valid_idx], prevalence, 1.10
        )
        candidate_prediction[valid_idx] = _rank_decode(
            mixed, prevalence, ratios
        )
        old = f1_score(targets[valid_idx], baseline_prediction[valid_idx],
                       average="macro", zero_division=0)
        new = f1_score(targets[valid_idx], candidate_prediction[valid_idx],
                       average="macro", zero_division=0)
        fold_rows.append({
            "fold": outer, "baseline_macro_f1": float(old),
            "candidate_macro_f1": float(new), "delta": float(new-old),
            "nonzero_paper_labels": int((weights > 0).sum()), "selection": audit,
        })
        fold_weights.append(weights); fold_ratios.append(ratios)
    baseline = float(f1_score(targets, baseline_prediction, average="macro", zero_division=0))
    candidate = float(f1_score(targets, candidate_prediction, average="macro", zero_division=0))
    bootstrap = _user_bootstrap(
        targets, baseline_prediction, candidate_prediction, groups,
        seed=373737, draws=3000,
    )
    production_weights = np.median(np.stack(fold_weights), axis=0)
    production_ratios = np.median(np.stack(fold_ratios), axis=0)
    production_mixed = ((1. - production_weights) * current_rank
                        + production_weights * paper_rank)
    production_base = _rank_decode(current_rank, targets.mean(0), 1.10)
    production_prediction = _rank_decode(
        production_mixed, targets.mean(0), production_ratios
    )
    production_baseline = float(f1_score(
        targets, production_base, average="macro", zero_division=0
    ))
    production = float(f1_score(
        targets, production_prediction, average="macro", zero_division=0
    ))
    auc_old = np.mean([
        roc_auc_score(targets[:, j], current[:, j])
        for j in range(config.NUM_FACTORS) if np.unique(targets[:, j]).size == 2
    ])
    auc_new = np.mean([
        roc_auc_score(targets[:, j], paper_oof[:, j])
        for j in range(config.NUM_FACTORS) if np.unique(targets[:, j]).size == 2
    ])
    adopted = bool(
        candidate >= baseline + .005 and bootstrap["positive_fraction"] >= .80
        and bootstrap["p05_delta"] >= 0 and production >= production_baseline
    )
    payload = {
        "training_version": TRAINING_VERSION,
        "evaluation": "five-fold user-disjoint OOF with nested per-label routing",
        "paper_dual_branch_oof_auc": float(auc_new),
        "current_ensemble_oof_auc": float(auc_old),
        "nested_baseline_macro_f1": baseline,
        "nested_candidate_macro_f1": candidate,
        "nested_delta": candidate - baseline,
        "production_baseline_oof_macro_f1": production_baseline,
        "production_candidate_oof_macro_f1": production,
        "production_delta": production - production_baseline,
        "production_nonzero_paper_labels": int((production_weights > 0).sum()),
        "user_cluster_bootstrap": bootstrap, "folds": fold_rows,
        "histories": histories, "adopted": adopted,
    }
    calibration = {
        "training_version": TRAINING_VERSION, "adopted": adopted,
        "weights": production_weights.tolist(), "ratios": production_ratios.tolist(),
        "training_prevalence": targets.mean(0).tolist(),
    }
    RESULTS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    CALIBRATION.write_text(json.dumps(calibration, indent=2), encoding="utf-8")
    np.savez_compressed(OOF, probabilities=paper_oof, targets=targets,
                        training_version=TRAINING_VERSION)
    print(json.dumps(payload, indent=2), flush=True)
    return payload


if __name__ == "__main__":
    cross_validate()
