"""Evidence-only refinement from the validated strict DeBERTa checkpoint.

Unlike the earlier Task1JointModel ablation, this experiment does not restart
from the generic Hugging Face weights.  It transfers the strong multitask
backbone and boundary head, adds a token-rationale head, and updates only the
top encoder layers.  Risk predictions remain the validated frozen ensemble.
"""
from __future__ import annotations

import json

import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from analyze_task1_evidence_v4 import _cue_cache, _decoder_grid, _evaluate
from baseline import _post_phrase_f1
from configs.config import config
from datasets.collator import SuicideRiskCollator
from datasets.dataset import SuicideRiskDataset
from inference.task1_evidence_v4 import (
    apply_evidence_policy, decode_model_evidence, load_evidence_calibration,
)
from models.backbone import DebertaBackbone
from models.heads import EvidenceExtractionHead
from models.losses import EvidenceLoss
from utils.seed import seed_everything
from utils.task1_metric import task1_score


OUTPUT = config.OUTPUT_DIR / "task1_evidence_refine_v9"
CHECKPOINT = OUTPUT / "model.pt"
RESULTS = OUTPUT / "results.json"
TRAINING_VERSION = "task1-evidence-refine-v9"


class EvidenceRefinementModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = DebertaBackbone()
        self.evidence_head = EvidenceExtractionHead()
        self.token_head = torch.nn.Sequential(
            torch.nn.Dropout(0.1), torch.nn.Linear(config.HIDDEN_SIZE, 1)
        )

    def forward(self, input_ids, attention_mask):
        hidden = self.backbone(input_ids, attention_mask).float()
        start, end = self.evidence_head(hidden)
        token = self.token_head(hidden).squeeze(-1)
        return {"start_logits": start, "end_logits": end, "token_logits": token}


def _initialise_from_validated(model):
    source = torch.load(config.OUTPUT_DIR / "best_model.pt", map_location="cpu")
    target = model.state_dict()
    compatible = {
        key: value for key, value in source.items()
        if key in target and target[key].shape == value.shape
    }
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    # Boundary weights already encode evidence salience. Their mean is a much
    # better token-head initialiser than random weights.
    with torch.no_grad():
        span = model.evidence_head.span_classifier
        token = model.token_head[1]
        token.weight.copy_(span.weight.mean(0, keepdim=True))
        token.bias.copy_(span.bias.mean().reshape(1))
    print(
        f"evidence-v9: transferred {len(compatible)} tensors; "
        f"new tensors={len(missing)}, unexpected={len(unexpected)}",
        flush=True,
    )


def _freeze_lower_layers(model, trainable_top_layers=4):
    for parameter in model.backbone.parameters():
        parameter.requires_grad = False
    layers = model.backbone.encoder.encoder.layer
    for layer in layers[-int(trainable_top_layers):]:
        for parameter in layer.parameters():
            parameter.requires_grad = True
    for parameter in model.evidence_head.parameters():
        parameter.requires_grad = True
    for parameter in model.token_head.parameters():
        parameter.requires_grad = True


def _loader(dataset, indices, shuffle):
    return DataLoader(
        Subset(dataset, indices), batch_size=config.BATCH_SIZE, shuffle=shuffle,
        collate_fn=SuicideRiskCollator(), num_workers=0,
        pin_memory=config.DEVICE == "cuda",
    )


def _move(batch, device):
    for key in ("input_ids", "attention_mask", "start_labels", "end_labels", "token_labels"):
        batch[key] = batch[key].to(device, non_blocking=True)
    return batch


class RefinementLoss(torch.nn.Module):
    def __init__(self, token_pos_weight):
        super().__init__()
        self.boundary = EvidenceLoss()
        self.register_buffer("token_pos_weight", torch.as_tensor(token_pos_weight).float())

    def forward(self, output, batch):
        boundary = self.boundary(
            output["start_logits"], output["end_logits"],
            batch["start_labels"], batch["end_labels"], batch["attention_mask"],
        )
        target = batch["token_labels"].float()
        mask = batch["attention_mask"].float()
        raw = torch.nn.functional.binary_cross_entropy_with_logits(
            output["token_logits"], target, reduction="none"
        )
        weight = torch.where(target > 0, self.token_pos_weight.to(raw.dtype), 1.0)
        bce = (raw * weight * mask).sum() / mask.sum().clamp_min(1.0)
        probability = torch.sigmoid(output["token_logits"]) * mask
        intersection = (probability * target).sum((1, 2))
        denominator = probability.sum((1, 2)) + (target * mask).sum((1, 2))
        dice = (1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
        # Boundary supervision preserves exact endpoints; dense token labels
        # teach the interior of multi-token evidence spans.
        total = 0.70 * boundary + 0.20 * bce + 0.10 * dice
        return total, {"boundary": boundary.detach(), "token_bce": bce.detach(), "dice": dice.detach()}


def _token_phrases(text, offsets, logits, threshold=0.55, max_tokens=12, limit=5):
    probability = torch.sigmoid(logits).detach().cpu().numpy()
    candidates = []
    for chunk, chunk_offsets in enumerate(offsets):
        valid = [i for i, (start, end) in enumerate(chunk_offsets) if end > start]
        active = {i for i in valid if probability[chunk, i] >= float(threshold)}
        cursor = 0
        while cursor < len(valid):
            token = valid[cursor]
            if token not in active:
                cursor += 1
                continue
            run = [token]; cursor += 1
            while (cursor < len(valid) and valid[cursor] == run[-1] + 1
                   and valid[cursor] in active and len(run) < int(max_tokens)):
                run.append(valid[cursor]); cursor += 1
            begin, finish = chunk_offsets[run[0]][0], chunk_offsets[run[-1]][1]
            phrase = text[begin:finish].strip()
            if phrase:
                candidates.append((float(probability[chunk, run].mean()), phrase))
    selected = []
    for _, phrase in sorted(candidates, reverse=True):
        normal = " ".join(phrase.casefold().split())
        if not any(normal in old or old in normal for _, old in selected):
            selected.append((phrase, normal))
        if len(selected) >= int(limit):
            break
    return [phrase for phrase, _ in selected]


@torch.no_grad()
def _collect(model, loader, device):
    model.eval(); rows = []
    for batch in tqdm(loader, desc="evidence-v9 validation", leave=False):
        metadata = batch; batch = _move(batch, device)
        output = model(batch["input_ids"], batch["attention_mask"])
        for i in range(len(metadata["row_id"])):
            rows.append({
                "row_id": str(metadata["row_id"][i]),
                "text": metadata["texts"][i],
                "offsets": metadata["offset_mappings"][i],
                "gold": metadata["evidences"][i],
                "start": output["start_logits"][i].cpu(),
                "end": output["end_logits"][i].cpu(),
                "token": output["token_logits"][i].cpu(),
            })
    return rows


def _evaluate_refined(rows, risk_by_id, calibration):
    settings = []
    for source in ("boundary", "token", "boundary_token"):
        for token_threshold in (0.45, 0.55, 0.65):
            scores = []
            for row in rows:
                risk = int(risk_by_id[row["row_id"]])
                boundary = decode_model_evidence(
                    row["text"], row["offsets"], row["start"], row["end"],
                    threshold=float(calibration["threshold"]),
                    max_tokens=int(calibration["max_tokens"]),
                    end_policy=calibration["end_policy"], limit=5,
                )
                token = _token_phrases(
                    row["text"], row["offsets"], row["token"],
                    threshold=token_threshold, max_tokens=12, limit=5,
                )
                if source == "boundary":
                    candidates = boundary
                elif source == "token":
                    candidates = token
                else:
                    candidates = boundary + token
                evidence = apply_evidence_policy(
                    row["text"], risk, candidates,
                    policy=calibration["cue_policy"], topk=int(calibration["topk"]),
                )
                scores.append(_post_phrase_f1(evidence, row["gold"]))
            settings.append({
                "source": source, "token_threshold": token_threshold,
                "phrase_f1": float(np.mean(scores)), "per_post": scores,
            })
    return sorted(settings, key=lambda row: row["phrase_f1"], reverse=True)


def train_task1_evidence_refine():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("Evidence refinement requires CUDA")
    seed_everything(config.SEED + 909)
    dataset = SuicideRiskDataset(config.CACHE_DIR / "train_cache.pt")
    labels = np.asarray([int(row["risk_label"]) for row in dataset.data])
    groups = np.asarray([str(row["anon_user_id"]) for row in dataset.data])
    train_idx, valid_idx = next(StratifiedGroupKFold(
        n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED
    ).split(np.zeros(len(dataset)), labels, groups))

    raw = torch.load(
        config.OUTPUT_DIR / "task1_evidence_v4" / "fold0_raw.pt",
        map_location="cpu", weights_only=False,
    )
    if not np.array_equal(np.asarray(raw["valid_idx"]), valid_idx):
        raise ValueError("V9 and V4 strict folds differ")
    risk_by_id = {str(row["row_id"]): int(row["risk"]) for row in raw["records"]}
    calibration = load_evidence_calibration()
    if calibration is None:
        raise FileNotFoundError("Adopted V4 evidence calibration is required")
    decoded = _decoder_grid(raw["records"])
    cues = _cue_cache(raw["records"])
    _, baseline_scores = _evaluate(
        raw["records"], decoded[(
            calibration["threshold"], calibration["max_tokens"], calibration["end_policy"]
        )], cues, np.arange(len(raw["records"])), calibration["cue_policy"], calibration["topk"],
    )

    positive = sum(float(dataset.data[int(i)]["token_labels"].sum()) for i in train_idx)
    total = sum(float(dataset.data[int(i)]["attention_mask"].sum()) for i in train_idx)
    token_weight = min(15.0, max(3.0, np.sqrt((total - positive) / max(positive, 1.0))))
    device = torch.device("cuda")
    model = EvidenceRefinementModel()
    _initialise_from_validated(model)
    _freeze_lower_layers(model, trainable_top_layers=4)
    model.to(device)
    criterion = RefinementLoss(token_weight).to(device)
    backbone_parameters = [
        parameter for name, parameter in model.named_parameters()
        if name.startswith("backbone.") and parameter.requires_grad
    ]
    head_parameters = [
        parameter for name, parameter in model.named_parameters()
        if not name.startswith("backbone.") and parameter.requires_grad
    ]
    optimizer = AdamW([
        {"params": backbone_parameters, "lr": 4e-6},
        {"params": head_parameters, "lr": 2e-5},
    ], weight_decay=config.WEIGHT_DECAY)
    train_loader = _loader(dataset, train_idx, True)
    valid_loader = _loader(dataset, valid_idx, False)
    epochs = 2
    updates = int(np.ceil(len(train_loader) / config.GRADIENT_ACCUMULATION)) * epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, max(1, int(updates * config.WARMUP_RATIO)), updates
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    history = []
    best_state = None; best_phrase = float(baseline_scores.mean())
    for epoch in range(1, epochs + 1):
        model.train(); optimizer.zero_grad(set_to_none=True); losses = []
        progress = tqdm(train_loader, desc=f"evidence-v9 epoch {epoch}/{epochs}")
        for step, batch in enumerate(progress, 1):
            batch = _move(batch, device)
            with torch.autocast(device_type="cuda", enabled=True):
                loss, parts = criterion(model(batch["input_ids"], batch["attention_mask"]), batch)
                scaled_loss = loss / config.GRADIENT_ACCUMULATION
            scaler.scale(scaled_loss).backward(); losses.append(float(loss.detach()))
            if step % config.GRADIENT_ACCUMULATION == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
                old_scale = scaler.get_scale(); scaler.step(optimizer); scaler.update()
                if scaler.get_scale() >= old_scale:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if step % 50 == 0:
                progress.set_postfix(loss=f"{np.mean(losses[-50:]):.3f}")
        rows = _collect(model, valid_loader, device)
        variants = _evaluate_refined(rows, risk_by_id, calibration)
        best = variants[0]
        history.append({
            "epoch": epoch, "train_loss": float(np.mean(losses)),
            "best_source": best["source"],
            "token_threshold": best["token_threshold"],
            "phrase_f1": best["phrase_f1"],
            "task1": task1_score(float(calibration["strict_risk_f1"]), best["phrase_f1"]),
            "variants": [{k: v for k, v in row.items() if k != "per_post"} for row in variants],
        })
        print(
            f"evidence-v9 epoch={epoch} loss={np.mean(losses):.4f} "
            f"source={best['source']} phrase_f1={best['phrase_f1']:.4f} "
            f"task1={history[-1]['task1']:.4f}", flush=True,
        )
        if best["phrase_f1"] > best_phrase:
            best_phrase = best["phrase_f1"]
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

    baseline_phrase = float(baseline_scores.mean())
    adopted = best_state is not None and best_phrase >= baseline_phrase + 0.005
    if adopted:
        torch.save(best_state, CHECKPOINT)
    payload = {
        "training_version": TRAINING_VERSION,
        "strict_train_posts": int(len(train_idx)), "strict_valid_posts": int(len(valid_idx)),
        "baseline": {
            "risk_f1": float(calibration["strict_risk_f1"]),
            "phrase_f1": baseline_phrase,
            "task1": task1_score(float(calibration["strict_risk_f1"]), baseline_phrase),
        },
        "epochs": history,
        "best_phrase_f1": best_phrase,
        "best_task1": task1_score(float(calibration["strict_risk_f1"]), best_phrase),
        "adopted": bool(adopted),
        "checkpoint": str(CHECKPOINT) if adopted else None,
    }
    RESULTS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    return payload


if __name__ == "__main__":
    train_task1_evidence_refine()
