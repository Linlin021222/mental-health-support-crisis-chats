"""Strict fold-0 test of paper-aligned end-to-end RF/PF joint learning."""
from __future__ import annotations

import json

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.optim import AdamW
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from configs.config import config
from datasets.factor_cache_builder import build_factor_cache
from inference.factor_nli import _rank_decode
from models.factor_joint_pfa_v45 import PFAJointFactorModel
from preprocess.preprocess import load_train_data
from trainer.factor_mhlat_v4 import _current_components
from trainer.factor_sentence_evidence_cv_v27 import _user_bootstrap
from trainer.factor_train import FactorDataset, WeightedGroupedASL, _loader
from utils.seed import seed_everything


OUTPUT = config.OUTPUT_DIR / "factor_joint_pfa_v45"
CHECKPOINT = OUTPUT / "fold0_model.pt"
PREDICTIONS = OUTPUT / "fold0_valid.npz"
RESULTS = OUTPUT / "fold0_results.json"
SOURCE = config.OUTPUT_DIR / "factor_cv" / "fold0_model.pt"
BASE_OOF = config.OUTPUT_DIR / "factor_cv" / "factor_oof_predictions.npz"
TRAINING_VERSION = "end-to-end-pfa-joint-multitask-v45"
EPOCHS = 2
FIXED_REPLACEMENT = .25


def _configure_trainable(model):
    """Tune new heads, accepted label heads and only the last two encoder layers."""
    for parameter in model.parameters():
        parameter.requires_grad = False
    new_prefixes = (
        "risk_query_delta", "protective_query_delta", "risk_branch.",
        "protective_branch.", "risk_factor_residual.",
        "protective_factor_residual.", "risk_factor_gate",
        "protective_factor_gate", "risk_level_head.", "task_log_vars",
    )
    accepted_heads = (
        "norm.", "label_queries", "label_weights", "label_bias",
        "global_risk.", "global_protective.",
    )
    for name, parameter in model.named_parameters():
        if name.startswith(new_prefixes + accepted_heads):
            parameter.requires_grad = True
        if name.startswith("encoder.encoder.layer.10.") or name.startswith(
            "encoder.encoder.layer.11."
        ):
            parameter.requires_grad = True


def _group_losses(loss_fn, logits, targets, counts):
    risk = loss_fn._loss(
        logits[:, :19], targets[:, :19], loss_fn.positive_weight[:19], counts[:, :19]
    )
    protective = loss_fn._loss(
        logits[:, 19:], targets[:, 19:], loss_fn.positive_weight[19:], counts[:, 19:]
    )
    return risk, protective


def _ordinal_risk_loss(logits, labels, class_weight, alpha=1.5):
    hard = F.cross_entropy(logits, labels, weight=class_weight)
    levels = torch.arange(4, device=logits.device).unsqueeze(0)
    distance = (levels - labels.unsqueeze(1)).abs().float()
    soft_target = torch.softmax(-alpha * distance, dim=1)
    ordinal = -(soft_target * F.log_softmax(logits, dim=1)).sum(1).mean()
    return .5 * hard + .5 * ordinal


def _joint_loss(losses, log_vars):
    # Eq. (18): each task learns its own homoscedastic uncertainty.  Clamping
    # prevents a tiny auxiliary dataset from suppressing a task completely.
    values = log_vars.clamp(-2.5, 2.5)
    return sum(torch.exp(-values[i]) * loss + values[i] for i, loss in enumerate(losses))


def _optimizer(model):
    encoder, accepted, new = [], [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("encoder."):
            encoder.append(parameter)
        elif name.startswith((
            "norm.", "label_queries", "label_weights", "label_bias",
            "global_risk.", "global_protective.",
        )):
            accepted.append(parameter)
        else:
            new.append(parameter)
    return AdamW([
        {"params": encoder, "lr": 1e-6},
        {"params": accepted, "lr": 5e-6},
        {"params": new, "lr": 2e-5},
    ], weight_decay=config.WEIGHT_DECAY)


def _train(model, loader, loss_fn, class_weight, device):
    optimizer = _optimizer(model)
    accumulation = config.GRADIENT_ACCUMULATION
    updates = EPOCHS * max(1, int(np.ceil(len(loader) / accumulation)))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, max(1, int(.08 * updates)), updates,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=config.FP16)
    history = []
    for epoch in range(1, EPOCHS + 1):
        model.train(); optimizer.zero_grad(set_to_none=True); rows = []
        for step, batch in enumerate(tqdm(loader, desc=f"V45 PFA joint epoch {epoch}/{EPOCHS}"), 1):
            targets = batch["factor_vectors"].to(device)
            counts = batch["factor_counts"].to(device)
            risk_labels = batch["risk_labels"].to(device)
            with torch.autocast(device_type=device.type, enabled=config.FP16):
                logits, aux = model(
                    batch["input_ids"].to(device), batch["attention_mask"].to(device),
                    return_aux=True,
                )
                risk_factor, protective_factor = _group_losses(
                    loss_fn, logits, targets, counts,
                )
                risk_level = _ordinal_risk_loss(
                    aux["risk_level_logits"], risk_labels, class_weight,
                )
                total = _joint_loss(
                    (risk_factor, protective_factor, risk_level), aux["task_log_vars"],
                ) / accumulation
            scaler.scale(total).backward()
            rows.append((float(risk_factor.detach()), float(protective_factor.detach()),
                         float(risk_level.detach())))
            if step % accumulation == 0 or step == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.,
                )
                old_scale = scaler.get_scale(); scaler.step(optimizer); scaler.update()
                if scaler.get_scale() >= old_scale:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
        mean = np.asarray(rows).mean(0)
        history.append({
            "epoch": epoch, "risk_factor_loss": float(mean[0]),
            "protective_factor_loss": float(mean[1]),
            "risk_level_loss": float(mean[2]),
            "task_log_vars": model.task_log_vars.detach().cpu().tolist(),
        })
        print(json.dumps(history[-1]), flush=True)
    return history


@torch.no_grad()
def _predict(model, loader, device):
    model.eval(); factors, risks = [], []
    for batch in tqdm(loader, desc="V45 strict validation"):
        with torch.autocast(device_type=device.type, enabled=config.FP16):
            logits, aux = model(
                batch["input_ids"].to(device), batch["attention_mask"].to(device),
                return_aux=True,
            )
        factors.append(torch.sigmoid(logits.float()).cpu().numpy())
        risks.append(torch.softmax(aux["risk_level_logits"].float(), -1).cpu().numpy())
    return np.vstack(factors).astype(np.float32), np.vstack(risks).astype(np.float32)


def _macro_auc(truth, probability):
    return float(np.mean([
        roc_auc_score(truth[:, label], probability[:, label])
        for label in range(24) if np.unique(truth[:, label]).size == 2
    ]))


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("V45 requires CUDA")
    OUTPUT.mkdir(parents=True, exist_ok=True); seed_everything(config.SEED + 4545)
    cache = build_factor_cache(train=True); dataset = FactorDataset(cache)
    frame = load_train_data().reset_index(drop=True)
    current, targets, calibration = _current_components()
    targets = targets.astype(np.float32)
    groups = frame.anon_user_id.astype(str).to_numpy()
    risk_labels = frame.risk_label.to_numpy(dtype=np.int64)
    train_idx, valid_idx = next(StratifiedGroupKFold(
        n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED,
    ).split(np.zeros(len(frame)), risk_labels, groups))
    device = torch.device("cuda")

    model = PFAJointFactorModel().to(device)
    incompatible = model.load_state_dict(
        torch.load(SOURCE, map_location="cpu", weights_only=True), strict=False,
    )
    allowed = (
        "risk_query_delta", "protective_query_delta", "risk_branch.",
        "protective_branch.", "risk_factor_residual.",
        "protective_factor_residual.", "risk_factor_gate",
        "protective_factor_gate", "risk_level_head.", "task_log_vars",
    )
    bad = [key for key in incompatible.missing_keys if not key.startswith(allowed)]
    if bad or incompatible.unexpected_keys:
        raise RuntimeError(f"Checkpoint mismatch: missing={bad}, unexpected={incompatible.unexpected_keys}")
    _configure_trainable(model)
    print(
        f"V45 trainable={sum(p.numel() for p in model.parameters() if p.requires_grad):,} "
        f"train={len(train_idx)} strict_valid={len(valid_idx)} user_overlap=0",
        flush=True,
    )
    positive = torch.tensor(targets[train_idx].sum(0), device=device)
    positive_weight = torch.sqrt(
        (len(train_idx) - positive) / positive.clamp_min(1.)
    ).clamp(1., 12.)
    loss_fn = WeightedGroupedASL(positive_weight).to(device)
    risk_support = np.bincount(risk_labels[train_idx], minlength=4)
    class_weight = torch.tensor(
        np.sqrt(len(train_idx) / np.maximum(risk_support, 1)),
        dtype=torch.float32, device=device,
    )
    history = _train(model, _loader(dataset, train_idx, True), loss_fn, class_weight, device)
    candidate, risk_probability = _predict(
        model, _loader(dataset, valid_idx, False), device,
    )
    torch.save(model.state_dict(), CHECKPOINT)
    np.savez_compressed(
        PREDICTIONS, valid_indices=valid_idx, probabilities=candidate,
        risk_probabilities=risk_probability, training_version=TRAINING_VERSION,
    )

    accepted = np.load(BASE_OOF)["semantic"].astype(np.float32)
    prevalence = targets[train_idx].mean(0)
    ratio = float(calibration["prevalence_ratio"])
    baseline_prediction = _rank_decode(current[valid_idx], prevalence, ratio)
    baseline = float(f1_score(
        targets[valid_idx], baseline_prediction, average="macro", zero_division=0,
    ))
    component_weight = (
        float(calibration["base_weight"]) * float(config.FACTOR_SEMANTIC_MODEL_WEIGHT)
    )
    grid = []
    for replacement in (0., .10, .15, .20, .25, .35, .50, 1.):
        probability = current[valid_idx] + component_weight * replacement * (
            candidate - accepted[valid_idx]
        )
        prediction = _rank_decode(probability, prevalence, ratio)
        grid.append({
            "replacement": replacement,
            "macro_f1": float(f1_score(
                targets[valid_idx], prediction, average="macro", zero_division=0,
            )),
        })
    fixed = next(row for row in grid if row["replacement"] == FIXED_REPLACEMENT)
    fixed_probability = current[valid_idx] + component_weight * FIXED_REPLACEMENT * (
        candidate - accepted[valid_idx]
    )
    fixed_prediction = _rank_decode(fixed_probability, prevalence, ratio)
    fixed["delta"] = fixed["macro_f1"] - baseline
    bootstrap = _user_bootstrap(
        targets[valid_idx], baseline_prediction, fixed_prediction, groups[valid_idx],
        seed=454545, draws=3000,
    )
    standalone = _rank_decode(candidate, prevalence, ratio)
    risk_prediction = risk_probability.argmax(1)
    per_group = {
        "risk_factor_macro_f1": float(f1_score(
            targets[valid_idx, :19], fixed_prediction[:, :19],
            average="macro", zero_division=0,
        )),
        "protective_factor_macro_f1": float(f1_score(
            targets[valid_idx, 19:], fixed_prediction[:, 19:],
            average="macro", zero_division=0,
        )),
    }
    payload = {
        "training_version": TRAINING_VERSION,
        "evaluation_scope": "strict user-disjoint fold0; fixed two-epoch continuation",
        "paper_transfer": {
            "independent_risk_protective_token_attention": True,
            "independent_risk_protective_mlp_heads": True,
            "ordinal_current_risk_auxiliary": True,
            "uncertainty_weighted_joint_loss": True,
            "last_two_encoder_layers_unfrozen": True,
            "next_post_dynamic_loss_omitted_as_task_mismatch": True,
        },
        "history": history,
        "trainable_parameters": int(sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )),
        "baseline_macro_f1": baseline,
        "candidate_standalone_macro_f1": float(f1_score(
            targets[valid_idx], standalone, average="macro", zero_division=0,
        )),
        "candidate_factor_auc": _macro_auc(targets[valid_idx], candidate),
        "auxiliary_risk_weighted_f1": float(f1_score(
            risk_labels[valid_idx], risk_prediction, average="weighted", zero_division=0,
        )),
        "fixed_25pct": fixed,
        "per_group_fixed": per_group,
        "grid_diagnostic": sorted(grid, key=lambda row: row["macro_f1"], reverse=True),
        "user_cluster_bootstrap": bootstrap,
        "promising": bool(
            fixed["delta"] >= .003 and bootstrap["positive_fraction"] >= .70
        ),
        "adopted": False,
    }
    RESULTS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    return payload


if __name__ == "__main__":
    main()
