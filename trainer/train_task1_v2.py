"""Strict user-disjoint ablation for ordinal risk + token rationale extraction."""
import json
import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from baseline import _apply_task1_rules, _post_phrase_f1
from configs.config import config
from datasets.cache_builder import build_cache
from datasets.collator import SuicideRiskCollator
from datasets.dataset import SuicideRiskDataset
from inference.predict import decode_evidence
from models.losses import EvidenceLoss, FactorLoss
from models.multitask_model_v2 import (
    SuicideRiskMultiTaskModelV2, ordinal_class_probabilities, v2_optimizer_parameters,
)
from utils.task1_metric import task1_score as competition_task1_score


CHECKPOINT = config.OUTPUT_DIR / "task1_v2_strict_model.pt"
RESULTS = config.OUTPUT_DIR / "task1_v2_strict_results.json"


def _loader(dataset, indices, shuffle):
    return DataLoader(
        Subset(dataset, indices), batch_size=config.BATCH_SIZE, shuffle=shuffle,
        collate_fn=SuicideRiskCollator(), num_workers=config.NUM_WORKERS,
        pin_memory=config.DEVICE == "cuda",
    )


class Task1V2Loss(torch.nn.Module):
    def __init__(self, class_weights, ordinal_pos_weight, token_pos_weight, factor_pos_weight):
        super().__init__()
        self.register_buffer("class_weights", class_weights)
        self.register_buffer("ordinal_pos_weight", ordinal_pos_weight)
        self.register_buffer("token_pos_weight", token_pos_weight)
        self.evidence = EvidenceLoss()
        self.factor = FactorLoss(factor_pos_weight)

    def forward(self, outputs, batch):
        risk = torch.nn.functional.cross_entropy(
            outputs["risk_logits"], batch["risk_labels"], weight=self.class_weights
        )
        thresholds = torch.arange(3, device=batch["risk_labels"].device)
        ordinal_targets = (batch["risk_labels"].unsqueeze(1) > thresholds).float()
        ordinal = torch.nn.functional.binary_cross_entropy_with_logits(
            outputs["ordinal_logits"], ordinal_targets,
            pos_weight=self.ordinal_pos_weight,
        )
        span = self.evidence(
            outputs["start_logits"], outputs["end_logits"],
            batch["start_labels"], batch["end_labels"], batch["attention_mask"],
        )
        mask = batch["attention_mask"].float()
        token_targets = batch["token_labels"].float()
        raw_token = torch.nn.functional.binary_cross_entropy_with_logits(
            outputs["token_logits"], token_targets, reduction="none"
        )
        token_weights = torch.where(
            token_targets > 0, self.token_pos_weight.to(raw_token.dtype), 1.0
        )
        token_bce = (raw_token * token_weights * mask).sum() / mask.sum().clamp_min(1.0)
        token_prob = torch.sigmoid(outputs["token_logits"]) * mask
        intersection = (token_prob * token_targets).sum(dim=(1, 2))
        denominator = token_prob.sum(dim=(1, 2)) + token_targets.sum(dim=(1, 2))
        token_dice = (1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
        token = 0.60 * token_bce + 0.40 * token_dice
        factor = self.factor(outputs["factor_logits"], batch["factor_vectors"])
        evidence = 0.60 * span + 0.40 * token
        total = risk + 0.35 * ordinal + evidence + factor
        return total, {"risk": risk, "ordinal": ordinal, "span": span, "token": token}


def _move(batch, device):
    for key in (
        "input_ids", "attention_mask", "risk_labels", "start_labels", "end_labels",
        "token_labels", "factor_vectors",
    ):
        batch[key] = batch[key].to(device)
    return batch


def decode_token_evidence(text, offsets, token_logits, threshold=0.5, max_tokens=12):
    probability = torch.sigmoid(token_logits).cpu().numpy()
    candidates = []
    for chunk, chunk_offsets in enumerate(offsets):
        active = [
            j for j, (a, b) in enumerate(chunk_offsets)
            if b > a and probability[chunk, j] >= threshold
        ]
        if not active:
            continue
        runs = [[active[0]]]
        for token in active[1:]:
            if token == runs[-1][-1] + 1 and len(runs[-1]) < max_tokens:
                runs[-1].append(token)
            else:
                runs.append([token])
        for run in runs:
            start, end = chunk_offsets[run[0]][0], chunk_offsets[run[-1]][1]
            phrase = text[start:end].strip()
            if phrase:
                candidates.append((float(probability[chunk, run].mean()), phrase))
    selected = []
    for _, phrase in sorted(candidates, reverse=True):
        normal = " ".join(phrase.casefold().split())
        if not any(normal in old.casefold() or old.casefold() in normal for old in selected):
            selected.append(phrase)
        if len(selected) == 5:
            break
    return selected


@torch.no_grad()
def _collect(model, loader, device):
    model.eval(); records = []
    for batch in loader:
        metadata = batch
        moved = _move(batch, device)
        output = model(moved["input_ids"], moved["attention_mask"])
        standard = torch.softmax(output["risk_logits"], dim=-1).cpu().numpy()
        ordinal = ordinal_class_probabilities(output["ordinal_logits"]).cpu().numpy()
        for i in range(len(metadata["row_id"])):
            records.append({
                "truth": int(moved["risk_labels"][i].cpu()),
                "standard": standard[i], "ordinal": ordinal[i],
                "text": metadata["texts"][i],
                "offsets": metadata["offset_mappings"][i],
                "gold": metadata["evidences"][i],
                "start": output["start_logits"][i].cpu(),
                "end": output["end_logits"][i].cpu(),
                "token": output["token_logits"][i].cpu(),
            })
    return records


def _score(records, ordinal_weight, token_threshold):
    truth, predictions, phrase_scores = [], [], []
    for item in records:
        probabilities = (
            (1.0 - ordinal_weight) * item["standard"]
            + ordinal_weight * item["ordinal"]
        )
        risk = int(np.argmax(probabilities))
        span_phrases = decode_evidence(
            item["text"], item["offsets"], item["start"], item["end"]
        )
        if token_threshold is None:
            evidence = span_phrases
        else:
            token_phrases = decode_token_evidence(
                item["text"], item["offsets"], item["token"], token_threshold, 12
            )
            evidence = token_phrases + [x for x in span_phrases if x not in token_phrases]
            evidence = evidence[:5]
        risk, evidence = _apply_task1_rules(item["text"], risk, evidence)
        truth.append(item["truth"]); predictions.append(risk)
        phrase_scores.append(_post_phrase_f1(evidence, item["gold"]))
    risk_f1 = f1_score(truth, predictions, average="weighted", zero_division=0)
    phrase_f1 = float(np.mean(phrase_scores))
    return risk_f1, phrase_f1, competition_task1_score(risk_f1, phrase_f1)


def train_task1_v2_strict():
    build_cache(train=True)
    dataset = SuicideRiskDataset(config.CACHE_DIR / "train_cache.pt")
    labels = np.asarray([int(x["risk_label"]) for x in dataset.data])
    groups = np.asarray([x["anon_user_id"] for x in dataset.data])
    split = StratifiedGroupKFold(
        n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED
    )
    train_idx, valid_idx = next(split.split(np.zeros(len(dataset)), labels, groups))
    train_loader = _loader(dataset, train_idx, True)
    valid_loader = _loader(dataset, valid_idx, False)
    device = torch.device(config.DEVICE)
    model = SuicideRiskMultiTaskModelV2().to(device)

    counts = np.bincount(labels[train_idx], minlength=4)
    class_weights = np.sqrt(len(train_idx) / np.maximum(counts, 1))
    class_weights = torch.tensor(class_weights / class_weights.mean(), dtype=torch.float, device=device)
    ordinal_targets = labels[train_idx, None] > np.arange(3)[None, :]
    ordinal_positive = ordinal_targets.sum(0)
    ordinal_weight = torch.tensor(
        np.sqrt((len(train_idx) - ordinal_positive) / np.maximum(ordinal_positive, 1)),
        dtype=torch.float, device=device,
    )
    factor_targets = torch.stack([dataset.data[int(i)]["factor_vector"] for i in train_idx]).float()
    factor_positive = factor_targets.sum(0)
    factor_weight = torch.sqrt((len(train_idx) - factor_positive) / factor_positive.clamp_min(1.0))
    factor_weight = factor_weight.clamp(1.0, 10.0).to(device)
    token_positive = sum(float(dataset.data[int(i)]["token_labels"].sum()) for i in train_idx)
    token_total = sum(float(dataset.data[int(i)]["attention_mask"].sum()) for i in train_idx)
    token_weight = torch.tensor(
        min(15.0, max(3.0, np.sqrt((token_total - token_positive) / max(token_positive, 1.0)))),
        device=device,
    )
    criterion = Task1V2Loss(class_weights, ordinal_weight, token_weight, factor_weight).to(device)
    optimizer = AdamW(v2_optimizer_parameters(model), weight_decay=config.WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=config.FP16)
    best = {"task1": -1.0}
    history = []
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Task1 V2 split train={len(train_idx)} valid={len(valid_idx)} users overlap=0")
    print(f"token positive weight={float(token_weight):.3f}")
    for epoch in range(1, config.EPOCHS + 1):
        model.train(); optimizer.zero_grad(set_to_none=True); losses = []
        for step, batch in enumerate(tqdm(train_loader, desc=f"task1-v2 epoch {epoch}"), 1):
            batch = _move(batch, device)
            with torch.autocast(device_type=device.type, enabled=config.FP16):
                output = model(batch["input_ids"], batch["attention_mask"])
                loss, _ = criterion(output, batch)
                loss = loss / config.GRADIENT_ACCUMULATION
            scaler.scale(loss).backward()
            losses.append(float(loss.detach()) * config.GRADIENT_ACCUMULATION)
            if step % config.GRADIENT_ACCUMULATION == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
        records = _collect(model, valid_loader, device)
        epoch_results = []
        for ordinal_blend in (0.0, 0.25, 0.50, 0.75, 1.0):
            for token_threshold in (None, 0.30, 0.40, 0.50, 0.60):
                risk_f1, phrase_f1, task1 = _score(records, ordinal_blend, token_threshold)
                epoch_results.append({
                    "ordinal_weight": ordinal_blend,
                    "token_threshold": token_threshold,
                    "risk_f1": risk_f1, "phrase_f1": phrase_f1, "task1": task1,
                })
        selected = max(epoch_results, key=lambda x: x["task1"])
        selected["epoch"] = epoch; selected["train_loss"] = float(np.mean(losses))
        history.append(selected)
        print("task1-v2", selected)
        if selected["task1"] > best["task1"]:
            best = dict(selected)
            torch.save(model.state_dict(), CHECKPOINT)
            RESULTS.write_text(json.dumps({"best": best, "history": history}, indent=2), encoding="utf-8")
    print("Best Task1 V2 strict:", best)
    return best


if __name__ == "__main__":
    train_task1_v2_strict()
