"""Leak-aware standalone MentalRoBERTa training for Subtask 2."""
import json
import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm

from configs.config import config
from datasets.factor_cache_builder import build_factor_cache
from models.factor_model import MentalRobertaFactorModel, factor_optimizer_parameters
from models.losses import AsymmetricLoss
from utils.factor_calibration import calibrate_factor_thresholds, save_factor_calibration


STRICT_CALIBRATION = config.OUTPUT_DIR / "factor_strict_calibration.json"
STRICT_CHECKPOINT = config.OUTPUT_DIR / "factor_strict_model.pt"
FULL_CHECKPOINT = config.OUTPUT_DIR / "factor_full_model.pt"


class FactorDataset(Dataset):
    def __init__(self, path):
        self.data = torch.load(path, map_location="cpu")
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        return self.data[index]


def _collate(rows):
    return {
        "input_ids": torch.stack([x["input_ids"] for x in rows]),
        "attention_mask": torch.stack([x["attention_mask"] for x in rows]),
        "factor_vectors": torch.stack([x["factor_vector"] for x in rows]),
        "factor_counts": torch.stack([
            x.get("factor_counts", x["factor_vector"]) for x in rows
        ]),
        "risk_labels": torch.tensor([
            int(x.get("risk_label", -100)) for x in rows
        ], dtype=torch.long),
    }


def _loader(dataset, indices, shuffle):
    return DataLoader(
        Subset(dataset, indices), batch_size=config.BATCH_SIZE, shuffle=shuffle,
        collate_fn=_collate, num_workers=config.NUM_WORKERS, pin_memory=config.DEVICE == "cuda",
    )


class WeightedGroupedASL(torch.nn.Module):
    """ASL plus positive weights; pure ASL was too weak for 8–16 positives."""
    def __init__(self, positive_weight):
        super().__init__()
        self.register_buffer("positive_weight", positive_weight)
        self.base = AsymmetricLoss()

    def _loss(self, logits, targets, positive_weight, factor_counts=None):
        p = torch.sigmoid(logits)
        n = 1.0 - p
        if config.ASL_CLIP:
            n = (n + config.ASL_CLIP).clamp(max=1.0)
        log_loss = targets * torch.log(p.clamp_min(1e-8))
        log_loss += (1.0 - targets) * torch.log(n.clamp_min(1e-8))
        pt = p * targets + n * (1.0 - targets)
        gamma = config.ASL_GAMMA_POS * targets + config.ASL_GAMMA_NEG * (1.0 - targets)
        focal = torch.pow(1.0 - pt, gamma)
        class_weight = 1.0 + (positive_weight.unsqueeze(0) - 1.0) * targets
        if factor_counts is not None and config.FACTOR_OCCURRENCE_ALPHA > 0:
            repeats = (factor_counts - 1.0).clamp_min(0.0)
            occurrence = 1.0 + config.FACTOR_OCCURRENCE_ALPHA * torch.log1p(repeats)
            occurrence = occurrence.clamp(max=2.5)
            class_weight = class_weight * (1.0 + (occurrence - 1.0) * targets)
        return -(log_loss * focal * class_weight).mean()

    def forward(self, logits, targets, factor_counts=None):
        risk_counts = None if factor_counts is None else factor_counts[:, :19]
        protective_counts = None if factor_counts is None else factor_counts[:, 19:]
        risk = self._loss(logits[:, :19], targets[:, :19], self.positive_weight[:19], risk_counts)
        protective = self._loss(
            logits[:, 19:], targets[:, 19:], self.positive_weight[19:], protective_counts
        )
        return 0.5 * (risk + protective)


class DistributionBalancedLoss(torch.nn.Module):
    """DB Loss adapted from Roche/BalancedLossNLP (EMNLP 2021).

    Parameters follow the repository's recommended DBloss configuration:
    re-balanced mapping (alpha=.1, beta=10, gamma=.9), focal gamma=2,
    and negative-tolerant regularisation (init_bias=.05, neg_scale=2).
    """
    def __init__(self, class_frequency, train_size):
        super().__init__()
        frequency = class_frequency.float().clamp_min(1.0)
        self.register_buffer("class_frequency", frequency)
        self.register_buffer("frequency_inverse", 1.0 / frequency)
        initial_bias = -torch.log(train_size / frequency - 1.0) * 0.05
        self.register_buffer("initial_bias", initial_bias)
        self.negative_scale = 2.0

    def forward(self, logits, targets):
        repeat_rate = (targets * self.frequency_inverse.unsqueeze(0)).sum(1, keepdim=True)
        positive_repeat = self.frequency_inverse.unsqueeze(0) / repeat_rate.clamp_min(1e-8)
        weight = torch.sigmoid(10.0 * (positive_repeat - 0.9)) + 0.1
        regularized_logits = logits + self.initial_bias.unsqueeze(0)
        regularized_logits = (
            regularized_logits * targets
            + regularized_logits * (1.0 - targets) * self.negative_scale
        )
        weight = weight * targets + weight * (1.0 - targets) / self.negative_scale
        raw = torch.nn.functional.binary_cross_entropy_with_logits(
            regularized_logits, targets, reduction="none"
        )
        probability_correct = torch.exp(-raw)
        weighted = torch.nn.functional.binary_cross_entropy_with_logits(
            regularized_logits, targets, weight=weight, reduction="none"
        )
        return (0.5 * (1.0 - probability_correct).pow(2.0) * weighted).mean()


@torch.no_grad()
def _probabilities(model, loader, device):
    model.eval()
    probabilities, targets = [], []
    for batch in loader:
        logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        probabilities.append(torch.sigmoid(logits).cpu().numpy())
        targets.append(batch["factor_vectors"].numpy())
    return np.vstack(probabilities), np.vstack(targets)


def _train_epoch(model, loader, loss_fn, optimizer, scaler, device, epoch):
    model.train(); optimizer.zero_grad(set_to_none=True); losses = []
    for step, batch in enumerate(tqdm(loader, desc=f"factor epoch {epoch}"), 1):
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        targets = batch["factor_vectors"].to(device)
        factor_counts = batch["factor_counts"].to(device)
        with torch.autocast(device_type=device.type, enabled=config.FP16):
            logits, semantic_logits = model(ids, mask, return_semantic=True)
            if isinstance(loss_fn, WeightedGroupedASL):
                classification_loss = loss_fn(logits, targets, factor_counts)
            else:
                classification_loss = loss_fn(logits, targets)
            semantic_loss = loss_fn(semantic_logits, targets)
            loss = (classification_loss + config.FACTOR_SEMANTIC_LOSS_WEIGHT * semantic_loss)
            loss = loss / config.GRADIENT_ACCUMULATION
        scaler.scale(loss).backward()
        losses.append(float(loss.detach()) * config.GRADIENT_ACCUMULATION)
        if step % config.GRADIENT_ACCUMULATION == 0 or step == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
            scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
    return float(np.mean(losses))


def _loss_and_model(dataset, train_indices, device):
    targets = torch.stack([dataset.data[int(i)]["factor_vector"] for i in train_indices]).float()
    positives = targets.sum(0)
    # Stronger than the former sqrt-only weighting, but capped to prevent a
    # handful of rare examples from destabilising the encoder.
    weights = torch.sqrt((len(train_indices) - positives) / positives.clamp_min(1.0))
    weights = weights.clamp(min=1.0, max=12.0).to(device)
    model = MentalRobertaFactorModel().to(device)
    if config.FACTOR_STANDALONE_LOSS == "distribution_balanced":
        loss = DistributionBalancedLoss(positives, len(train_indices))
    else:
        loss = WeightedGroupedASL(weights)
    return model, loss.to(device)


def train_factor_strict():
    cache = build_factor_cache(train=True)
    dataset = FactorDataset(cache)
    risk = np.asarray([x["risk_label"] for x in dataset.data])
    groups = np.asarray([x["anon_user_id"] for x in dataset.data])
    outer = StratifiedGroupKFold(n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED)
    outer_train, outer_valid = next(outer.split(np.zeros(len(dataset)), risk, groups))
    inner = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=config.SEED + 91)
    fit_rel, calibration_rel = next(inner.split(outer_train, groups=groups[outer_train]))
    fit_idx, calibration_idx = outer_train[fit_rel], outer_train[calibration_rel]
    print(f"Task 2 split: fit={len(fit_idx)}, calibration={len(calibration_idx)}, "
          f"strict_test={len(outer_valid)}; user overlap=0; "
          f"loss={config.FACTOR_STANDALONE_LOSS}")
    device = torch.device(config.DEVICE)
    model, loss_fn = _loss_and_model(dataset, fit_idx, device)
    optimizer = AdamW(factor_optimizer_parameters(model), weight_decay=config.WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=config.FP16)
    fit_loader = _loader(dataset, fit_idx, True)
    calibration_loader = _loader(dataset, calibration_idx, False)
    valid_loader = _loader(dataset, outer_valid, False)
    best, best_epoch = -1.0, 1
    fit_targets = np.vstack([dataset.data[int(i)]["factor_vector"].numpy() for i in fit_idx])
    fit_prevalence = fit_targets.mean(axis=0)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, config.FACTOR_EPOCHS + 1):
        train_loss = _train_epoch(model, fit_loader, loss_fn, optimizer, scaler, device, epoch)
        calibration_probs, calibration_targets = _probabilities(model, calibration_loader, device)
        thresholds = calibrate_factor_thresholds(
            calibration_targets, calibration_probs, reference_prevalence=fit_prevalence
        )
        valid_probs, valid_targets = _probabilities(model, valid_loader, device)
        predictions = valid_probs >= thresholds[None, :]
        score = f1_score(valid_targets, predictions, average="macro", zero_division=0)
        print(f"factor epoch={epoch} train_loss={train_loss:.4f} strict_task2={score:.4f}")
        if score > best:
            best, best_epoch = score, epoch
            torch.save(model.state_dict(), STRICT_CHECKPOINT)
            save_factor_calibration(thresholds, calibration_targets, STRICT_CALIBRATION)
            details = {
                config.ID2FACTOR[j]: {
                    "f1": float(f1_score(valid_targets[:, j], predictions[:, j], zero_division=0)),
                    "gold_support": int(valid_targets[:, j].sum()),
                    "predicted_positive": int(predictions[:, j].sum()),
                    "threshold": float(thresholds[j]),
                } for j in range(config.NUM_FACTORS)
            }
            (config.OUTPUT_DIR / "mentalroberta_task2_per_label.json").write_text(
                json.dumps(details, indent=2), encoding="utf-8"
            )
            payload = json.loads(STRICT_CALIBRATION.read_text(encoding="utf-8"))
            payload["best_epoch"] = best_epoch
            payload["strict_task2"] = float(best)
            STRICT_CALIBRATION.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Best standalone MentalRoBERTa strict Task 2: {best:.4f} at epoch {best_epoch}")
    return best


def train_factor_full():
    if not STRICT_CALIBRATION.exists():
        raise FileNotFoundError("Run --mode factor-strict first to obtain leak-free thresholds.")
    cache = build_factor_cache(train=True)
    dataset = FactorDataset(cache)
    all_idx = np.arange(len(dataset))
    device = torch.device(config.DEVICE)
    model, loss_fn = _loss_and_model(dataset, all_idx, device)
    optimizer = AdamW(factor_optimizer_parameters(model), weight_decay=config.WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=config.FP16)
    loader = _loader(dataset, all_idx, True)
    strict_payload = json.loads(STRICT_CALIBRATION.read_text(encoding="utf-8"))
    epochs = int(strict_payload.get("best_epoch", 3))
    for epoch in range(1, epochs + 1):
        loss = _train_epoch(model, loader, loss_fn, optimizer, scaler, device, epoch)
        print(f"full factor epoch={epoch} train_loss={loss:.4f}")
    torch.save(model.state_dict(), FULL_CHECKPOINT)
    thresholds = np.asarray([strict_payload["thresholds"][x] for x in config.FACTOR_LABELS])
    targets = np.vstack([x["factor_vector"].numpy() for x in dataset.data])
    save_factor_calibration(thresholds, targets)
    print(f"Full MentalRoBERTa factor checkpoint: {FULL_CHECKPOINT}")
    return FULL_CHECKPOINT
