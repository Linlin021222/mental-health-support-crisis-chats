"""A leak-free training entry point for the three competition targets."""
import numpy as np
import json
import torch
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score
from tqdm import tqdm
from configs.config import config
from datasets.cache_builder import build_cache
from datasets.dataset import SuicideRiskDataset
from datasets.collator import SuicideRiskCollator
from models.multitask_model import SuicideRiskMultiTaskModel, get_optimizer_parameters
from models.losses import MultiTaskLoss
from inference.predict import decode_evidence
from inference.task1_evidence_v4 import (
    apply_evidence_policy, correct_risk_only, decode_model_evidence,
    load_evidence_calibration,
)
from baseline import _apply_task1_rules, _post_phrase_f1
from utils.factor_calibration import (
    calibrate_factor_thresholds, save_factor_calibration, blend_cpu_factor_probabilities,
)
from utils.task1_metric import task1_score as competition_task1_score
from utils.task1_metric import composite_score as competition_composite_score


def _loader(dataset, indices, shuffle):
    return DataLoader(Subset(dataset, indices), batch_size=config.BATCH_SIZE, shuffle=shuffle,
                      collate_fn=SuicideRiskCollator(), num_workers=config.NUM_WORKERS,
                      pin_memory=config.DEVICE == "cuda")


def _move(batch, device):
    return {key: batch[key].to(device) for key in ("input_ids", "attention_mask", "risk_labels",
            "start_labels", "end_labels", "factor_vectors")}


def _evaluate(model, loader, criterion, device, factor_thresholds=None):
    model.eval(); losses = []; truth = []; pred = []; phrase_scores = []
    factor_truth = []; factor_pred = []
    evidence_v4 = load_evidence_calibration()
    with torch.no_grad():
        for batch in loader:
            metadata = batch
            tensors = _move(batch, device)
            output = model(tensors["input_ids"], tensors["attention_mask"])
            losses.append(float(criterion(output, tensors)["loss"]))
            raw_risks = output["risk_logits"].argmax(-1).cpu().tolist()
            factor_truth.append(tensors["factor_vectors"].cpu().numpy())
            factor_pred.append(torch.sigmoid(output["factor_logits"]).cpu().numpy())
            for i, raw_risk in enumerate(raw_risks):
                if evidence_v4 is None:
                    evidence = decode_evidence(
                        metadata["texts"][i], metadata["offset_mappings"][i],
                        output["start_logits"][i], output["end_logits"][i]
                    )
                    risk, evidence = _apply_task1_rules(
                        metadata["texts"][i], raw_risk, evidence
                    )
                else:
                    evidence = decode_model_evidence(
                        metadata["texts"][i], metadata["offset_mappings"][i],
                        output["start_logits"][i], output["end_logits"][i],
                        threshold=float(evidence_v4["threshold"]),
                        max_tokens=int(evidence_v4["max_tokens"]),
                        end_policy=str(evidence_v4["end_policy"]), limit=5,
                    )
                    risk = correct_risk_only(metadata["texts"][i], raw_risk)
                    evidence = apply_evidence_policy(
                        metadata["texts"][i], risk, evidence,
                        policy=str(evidence_v4["cue_policy"]),
                        topk=int(evidence_v4["topk"]),
                    )
                truth.append(int(tensors["risk_labels"][i].cpu()))
                pred.append(risk)
                phrase_scores.append(_post_phrase_f1(evidence, metadata["evidences"][i]))
    risk_f1 = f1_score(truth, pred, average="weighted")
    phrase_f1 = float(np.mean(phrase_scores))
    task1 = competition_task1_score(risk_f1, phrase_f1)
    thresholds = (np.full(config.NUM_FACTORS, config.FACTOR_THRESHOLD, dtype=np.float32)
                  if factor_thresholds is None else np.asarray(factor_thresholds, dtype=np.float32))
    factor_truth = np.vstack(factor_truth)
    factor_binary = np.vstack(factor_pred) >= thresholds[None, :]
    task2 = f1_score(
        factor_truth, factor_binary,
        average="macro", zero_division=0,
    )
    composite = competition_composite_score(risk_f1, phrase_f1, task2)
    per_label = f1_score(factor_truth, factor_binary, average=None, zero_division=0)
    factor_details = {
        config.ID2FACTOR[j]: {
            "f1": float(per_label[j]),
            "gold_support": int(factor_truth[:, j].sum()),
            "predicted_positive": int(factor_binary[:, j].sum()),
            "threshold": float(thresholds[j]),
        }
        for j in range(config.NUM_FACTORS)
    }
    return float(np.mean(losses)), risk_f1, phrase_f1, task1, task2, composite, factor_details


@torch.no_grad()
def _fit_factor_thresholds(model, loader, device):
    """Fit thresholds on training predictions, never on the outer holdout."""
    model.eval()
    probabilities, targets = [], []
    for batch in loader:
        tensors = _move(batch, device)
        output = model(tensors["input_ids"], tensors["attention_mask"])
        probabilities.append(torch.sigmoid(output["factor_logits"]).cpu().numpy())
        targets.append(tensors["factor_vectors"].cpu().numpy())
    probabilities, targets = np.vstack(probabilities), np.vstack(targets)
    return calibrate_factor_thresholds(targets, probabilities), targets


def train():
    cache_file = config.CACHE_DIR / "train_cache.pt"
    if not cache_file.exists():
        build_cache(train=True)
    dataset = SuicideRiskDataset(cache_file)
    cached_shape = tuple(dataset.data[0]["input_ids"].shape)
    expected_shape = (config.MAX_CHUNKS, config.MAX_LENGTH)
    if cached_shape != expected_shape or "anon_user_id" not in dataset.data[0]:
        print(f"Rebuilding stale cache {cached_shape}; expected {expected_shape}")
        build_cache(train=True)
        dataset = SuicideRiskDataset(cache_file)
    labels = [int(item["risk_label"]) for item in dataset.data]
    groups = [item["anon_user_id"] for item in dataset.data]
    counts = np.bincount(labels, minlength=config.NUM_RISK_CLASSES)
    n_splits = min(config.N_FOLDS, int(counts.min()))
    if n_splits < 2:
        raise ValueError("Each risk class needs at least two examples for validation.")
    folds = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=config.SEED)
    train_idx, valid_idx = next(folds.split(np.zeros(len(labels)), labels, groups=groups))
    train_loader, valid_loader = _loader(dataset, train_idx, True), _loader(dataset, valid_idx, False)
    device = torch.device(config.DEVICE)
    model = SuicideRiskMultiTaskModel().to(device)
    optimizer = AdamW(get_optimizer_parameters(model), weight_decay=config.WEIGHT_DECAY)
    train_counts = np.bincount(np.asarray(labels)[train_idx], minlength=config.NUM_RISK_CLASSES)
    class_weights = np.sqrt(len(train_idx) / np.maximum(train_counts, 1))
    class_weights = torch.tensor(class_weights / class_weights.mean(), dtype=torch.float, device=device)
    train_factor_labels = torch.stack([dataset.data[int(i)]["factor_vector"] for i in train_idx]).float()
    factor_positive = train_factor_labels.sum(0)
    factor_pos_weight = torch.sqrt((len(train_idx) - factor_positive) / factor_positive.clamp_min(1.0))
    factor_pos_weight = factor_pos_weight.clamp(min=1.0, max=10.0).to(device)
    criterion = MultiTaskLoss(risk_class_weights=class_weights,
                              factor_pos_weight=factor_pos_weight).to(device)
    scaler = torch.amp.GradScaler("cuda", enabled=config.FP16)
    # This checkpoint is consumed by Task 1 inference.  Task 2 has its own
    # MentalRoBERTa pipeline, so checkpoint selection must follow Task 1.
    best_score = -1.0
    output_dir = config.OUTPUT_DIR / "fold0"; output_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, config.EPOCHS + 1):
        model.train(); optimizer.zero_grad(set_to_none=True); running = []
        for step, batch in enumerate(tqdm(train_loader, desc=f"epoch {epoch}"), 1):
            batch = _move(batch, device)
            with torch.autocast(device_type=device.type, enabled=config.FP16):
                output = model(batch["input_ids"], batch["attention_mask"])
                loss = criterion(output, batch)["loss"] / config.GRADIENT_ACCUMULATION
            scaler.scale(loss).backward(); running.append(float(loss.detach()) * config.GRADIENT_ACCUMULATION)
            if step % config.GRADIENT_ACCUMULATION == 0 or step == len(train_loader):
                scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
        factor_thresholds, threshold_targets = _fit_factor_thresholds(model, train_loader, device)
        (valid_loss, valid_risk_f1, valid_phrase_f1, valid_task1, valid_task2,
         valid_composite, factor_details) = _evaluate(
            model, valid_loader, criterion, device, factor_thresholds
        )
        print(f"epoch={epoch} train_loss={np.mean(running):.4f} valid_loss={valid_loss:.4f} "
              f"risk_f1={valid_risk_f1:.4f} phrase_f1={valid_phrase_f1:.4f} task1={valid_task1:.4f} "
              f"task2={valid_task2:.4f} composite={valid_composite:.4f}")
        if valid_task1 > best_score:
            best_score = valid_task1
            torch.save(model.state_dict(), output_dir / "best_model.pt")
            torch.save(model.state_dict(), config.OUTPUT_DIR / "best_model.pt")
            save_factor_calibration(factor_thresholds, threshold_targets)
            (config.OUTPUT_DIR / "strict_task2_per_label.json").write_text(
                json.dumps(factor_details, indent=2), encoding="utf-8"
            )
    print(f"Best strict user-holdout Task 1 score: {best_score:.4f}")


def train_full():
    """Fit the final multi-task model on every labelled training post.

    This is the submission-training path.  It deliberately performs no
    holdout split: the epoch count has already been selected by the strict
    user-disjoint experiment, so all 1,635 labelled posts can now contribute
    to the final checkpoint.  Task 2 keeps the MLP factor head and the same
    prevalence-aware positive weights used during strict training.
    """
    cache_file = config.CACHE_DIR / "train_cache.pt"
    if not cache_file.exists():
        build_cache(train=True)
    dataset = SuicideRiskDataset(cache_file)
    cached_shape = tuple(dataset.data[0]["input_ids"].shape)
    expected_shape = (config.MAX_CHUNKS, config.MAX_LENGTH)
    if cached_shape != expected_shape:
        print(f"Rebuilding stale cache {cached_shape}; expected {expected_shape}")
        build_cache(train=True)
        dataset = SuicideRiskDataset(cache_file)

    all_idx = np.arange(len(dataset))
    loader = _loader(dataset, all_idx, True)
    device = torch.device(config.DEVICE)
    model = SuicideRiskMultiTaskModel().to(device)
    optimizer = AdamW(get_optimizer_parameters(model), weight_decay=config.WEIGHT_DECAY)

    labels = np.asarray([int(item["risk_label"]) for item in dataset.data])
    counts = np.bincount(labels, minlength=config.NUM_RISK_CLASSES)
    class_weights = np.sqrt(len(dataset) / np.maximum(counts, 1))
    class_weights = torch.tensor(class_weights / class_weights.mean(), dtype=torch.float, device=device)

    factor_labels = torch.stack([item["factor_vector"] for item in dataset.data]).float()
    factor_positive = factor_labels.sum(0)
    factor_pos_weight = torch.sqrt((len(dataset) - factor_positive) / factor_positive.clamp_min(1.0))
    factor_pos_weight = factor_pos_weight.clamp(min=1.0, max=10.0).to(device)
    criterion = MultiTaskLoss(
        risk_class_weights=class_weights,
        factor_pos_weight=factor_pos_weight,
    ).to(device)
    scaler = torch.amp.GradScaler("cuda", enabled=config.FP16)

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = config.OUTPUT_DIR / "full_train_model.pt"
    epochs = config.FULL_TRAIN_EPOCHS
    print(f"Final training: {len(dataset)} posts, no holdout, {epochs} epochs")
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running = []
        for step, batch in enumerate(tqdm(loader, desc=f"full epoch {epoch}"), 1):
            batch = _move(batch, device)
            with torch.autocast(device_type=device.type, enabled=config.FP16):
                output = model(batch["input_ids"], batch["attention_mask"])
                loss = criterion(output, batch)["loss"] / config.GRADIENT_ACCUMULATION
            scaler.scale(loss).backward()
            running.append(float(loss.detach()) * config.GRADIENT_ACCUMULATION)
            if step % config.GRADIENT_ACCUMULATION == 0 or step == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        torch.save(model.state_dict(), checkpoint)
        print(f"full epoch={epoch} train_loss={np.mean(running):.4f} saved={checkpoint}")
    model.eval()
    calibration_loader = _loader(dataset, all_idx, False)
    factor_probabilities, factor_targets = [], []
    with torch.no_grad():
        for batch in tqdm(calibration_loader, desc="calibrating Task 2"):
            tensors = _move(batch, device)
            output = model(tensors["input_ids"], tensors["attention_mask"])
            factor_probabilities.append(torch.sigmoid(output["factor_logits"]).cpu().numpy())
            factor_targets.append(tensors["factor_vectors"].cpu().numpy())
    factor_probabilities = np.vstack(factor_probabilities)
    factor_targets = np.vstack(factor_targets)
    factor_probabilities = blend_cpu_factor_probabilities(
        [item["text"] for item in dataset.data], factor_probabilities
    )
    thresholds = calibrate_factor_thresholds(factor_targets, factor_probabilities)
    calibration_file = save_factor_calibration(thresholds, factor_targets)
    calibrated_f1 = f1_score(
        factor_targets,
        factor_probabilities >= thresholds[None, :],
        average="macro",
        zero_division=0,
    )
    print(f"Task 2 calibration-fit F1={calibrated_f1:.4f}; saved={calibration_file}")
    print(f"Full-data checkpoint ready: {checkpoint}")
    return checkpoint


if __name__ == "__main__":
    train()
