"""Count-aware continuation of the accepted Task-2 encoder (V9).

The workbook repeats factor names when a factor has multiple mentions.  V9
preserves the binary objective while predicting log(1 + occurrence count) as
an auxiliary label-specific task.  This uses richer annotation without
changing the competition's binary output format.
"""
from __future__ import annotations

import json
import math

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.optim import AdamW
from tqdm import tqdm

from configs.config import config
from datasets.factor_cache_builder import build_factor_cache
from inference.factor_nli import _rank_decode
from models.factor_model import MentalRobertaFactorModel
from preprocess.preprocess import load_train_data
from trainer.factor_llm_lexical_v6 import _current_v3_probability
from trainer.factor_train import FactorDataset, WeightedGroupedASL, _loader
from utils.seed import seed_everything


OUTPUT = config.OUTPUT_DIR / "factor_count_aux_v9"
CHECKPOINT = OUTPUT / "fold0_model.pt"
RESULTS_FILE = OUTPUT / "fold0_results.json"
TRAINING_VERSION = "factor-occurrence-auxiliary-v9"
COUNT_WEIGHT = 0.15
EPOCHS = 2
SEED = 909090


class CountAwareFactorModel(MentalRobertaFactorModel):
    def __init__(self):
        # The fold checkpoint immediately restores all inherited tensors.
        super().__init__(initialise_labels=False)
        hidden = self.encoder.config.hidden_size
        self.count_weights = nn.Parameter(torch.empty(config.NUM_FACTORS, hidden))
        self.count_bias = nn.Parameter(torch.zeros(config.NUM_FACTORS))
        nn.init.xavier_uniform_(self.count_weights)

    def forward(self, input_ids, attention_mask, return_auxiliary=False):
        batch, chunks, length = input_ids.shape
        flat_ids = input_ids.reshape(batch * chunks, length)
        flat_mask = attention_mask.reshape(batch * chunks, length)
        hidden = self.encoder(input_ids=flat_ids, attention_mask=flat_mask).last_hidden_state
        hidden = hidden.float().reshape(batch, chunks * length, -1)
        mask = attention_mask.reshape(batch, chunks * length).bool()
        tokens = self.norm(hidden)
        scores = torch.einsum("bth,kh->btk", tokens, self.label_queries)
        scores = scores / math.sqrt(tokens.size(-1))
        scores = scores.masked_fill(~mask.unsqueeze(-1), -1e4)
        attention = torch.softmax(scores, dim=1)
        label_repr = torch.einsum("btk,bth->bkh", attention, tokens)
        dropped = self.dropout(label_repr)
        local = (dropped * self.label_weights.unsqueeze(0)).sum(-1) + self.label_bias
        mask_float = mask.unsqueeze(-1).to(tokens.dtype)
        global_repr = (tokens * mask_float).sum(1) / mask_float.sum(1).clamp_min(1.0)
        global_logits = torch.cat(
            [self.global_risk(global_repr), self.global_protective(global_repr)], dim=-1
        )
        logits = local + global_logits
        if not return_auxiliary:
            return logits
        document_semantic = torch.nn.functional.normalize(global_repr, dim=-1)
        label_semantic = torch.nn.functional.normalize(self.label_queries, dim=-1)
        semantic_logits = torch.einsum("bh,kh->bk", document_semantic, label_semantic)
        semantic_logits = semantic_logits / config.FACTOR_SEMANTIC_TEMPERATURE
        count_logits = (dropped * self.count_weights.unsqueeze(0)).sum(-1) + self.count_bias
        return logits, semantic_logits, count_logits


def _count_loss(logits, counts):
    target = torch.log1p(counts)
    prediction = torch.nn.functional.softplus(logits)
    raw = torch.nn.functional.smooth_l1_loss(prediction, target, reduction="none")
    positive = counts > 0
    positive_loss = raw[positive].mean() if positive.any() else raw.new_zeros(())
    negative_loss = raw[~positive].mean() if (~positive).any() else raw.new_zeros(())
    return 0.5 * (positive_loss + negative_loss)


@torch.no_grad()
def _predict(model, loader, device):
    model.eval(); rows = []
    for batch in loader:
        logits = model(
            batch["input_ids"].to(device), batch["attention_mask"].to(device)
        )
        rows.append(torch.sigmoid(logits).cpu().numpy())
    return np.vstack(rows)


def _v3_with_replaced_semantic(new_semantic, valid_idx):
    calibration = json.loads(
        (config.OUTPUT_DIR / "factor_cross_encoder_v2" / "calibration.json").read_text(
            encoding="utf-8"
        )
    )
    base_saved = np.load(config.OUTPUT_DIR / "factor_cv" / "factor_oof_predictions.npz")
    cpu = base_saved["cpu"][valid_idx]
    old = np.load(
        config.OUTPUT_DIR / "factor_cross_encoder" / "oof_predictions.npz"
    )["probabilities"][valid_idx]
    prototype = np.load(
        config.OUTPUT_DIR / "factor_cross_encoder_v2" / "oof_predictions.npz"
    )["probabilities"][valid_idx]
    base = .7 * new_semantic + .3 * cpu
    return (
        float(calibration["base_weight"]) * base
        + float(calibration["old_cross_weight"]) * old
        + float(calibration["new_cross_weight"]) * prototype
    ).astype(np.float32)


def train_fold0():
    if not torch.cuda.is_available():
        raise RuntimeError("Count-aware V9 requires CUDA")
    OUTPUT.mkdir(parents=True, exist_ok=True); seed_everything(SEED)
    cache = build_factor_cache(train=True)
    dataset = FactorDataset(cache)
    frame = load_train_data().reset_index(drop=True)
    targets = np.vstack(frame.factor_vector.to_numpy()).astype(np.int8)
    risk = frame.risk_label.to_numpy(np.int64)
    groups = frame.anon_user_id.astype(str).to_numpy()
    train_idx, valid_idx = next(StratifiedGroupKFold(
        n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED,
    ).split(np.zeros(len(frame)), risk, groups))
    device = torch.device("cuda")
    model = CountAwareFactorModel().to(device)
    source = config.OUTPUT_DIR / "factor_cv" / "fold0_model.pt"
    missing, unexpected = model.load_state_dict(
        torch.load(source, map_location="cpu"), strict=False
    )
    allowed = {"count_weights", "count_bias"}
    if set(missing) != allowed or unexpected:
        raise RuntimeError(f"Unexpected checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    # Initialise the occurrence reader from the already learned binary reader.
    with torch.no_grad():
        model.count_weights.copy_(model.label_weights)
        model.count_bias.copy_(model.label_bias)
    positive = torch.from_numpy(targets[train_idx].sum(0).astype(np.float32))
    class_weight = torch.sqrt((len(train_idx) - positive) / positive.clamp_min(1.)).clamp(1., 12.)
    loss_fn = WeightedGroupedASL(class_weight.to(device))
    backbone, heads = [], []
    for name, parameter in model.named_parameters():
        (backbone if name.startswith("encoder.") else heads).append(parameter)
    optimizer = AdamW([
        {"params": backbone, "lr": config.BACKBONE_LR * .35},
        {"params": heads, "lr": config.HEAD_LR * .50},
    ], weight_decay=config.WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=config.FP16)
    train_loader = _loader(dataset, train_idx, True)
    valid_loader = _loader(dataset, valid_idx, False)
    prevalence = targets[train_idx].mean(0)
    old_probability, _ = _current_v3_probability()
    old_prediction = _rank_decode(old_probability[valid_idx], prevalence, 1.10)
    baseline = float(f1_score(
        targets[valid_idx], old_prediction, average="macro", zero_division=0
    ))
    epochs = [{"epoch": 0, "candidate_macro_f1": baseline, "baseline_macro_f1": baseline}]
    best = baseline
    for epoch in range(1, EPOCHS + 1):
        model.train(); optimizer.zero_grad(set_to_none=True); losses = []
        for step, batch in enumerate(tqdm(train_loader, desc=f"Factor count V9 epoch {epoch}"), 1):
            ids = batch["input_ids"].to(device); mask = batch["attention_mask"].to(device)
            binary = batch["factor_vectors"].to(device); counts = batch["factor_counts"].to(device)
            with torch.autocast(device_type="cuda", enabled=config.FP16):
                logits, semantic, count_logits = model(ids, mask, return_auxiliary=True)
                classification = loss_fn(logits, binary, counts)
                semantic_loss = loss_fn(semantic, binary)
                occurrence = _count_loss(count_logits, counts)
                loss = classification + config.FACTOR_SEMANTIC_LOSS_WEIGHT * semantic_loss
                loss = loss + COUNT_WEIGHT * occurrence
                loss = loss / config.GRADIENT_ACCUMULATION
            scaler.scale(loss).backward(); losses.append(float(loss.detach()) * config.GRADIENT_ACCUMULATION)
            if step % config.GRADIENT_ACCUMULATION == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
        new_semantic = _predict(model, valid_loader, device)
        candidate_probability = _v3_with_replaced_semantic(new_semantic, valid_idx)
        candidate_prediction = _rank_decode(candidate_probability, prevalence, 1.10)
        score = float(f1_score(
            targets[valid_idx], candidate_prediction, average="macro", zero_division=0
        ))
        standalone = float(f1_score(
            targets[valid_idx], _rank_decode(new_semantic, prevalence, 1.10),
            average="macro", zero_division=0,
        ))
        row = {
            "epoch": epoch, "train_loss": float(np.mean(losses)),
            "standalone_macro_f1": standalone,
            "baseline_macro_f1": baseline, "candidate_macro_f1": score,
        }
        epochs.append(row); print(json.dumps(row, indent=2), flush=True)
        if score > best:
            best = score; torch.save(model.state_dict(), CHECKPOINT)
    adopted = bool(best >= baseline + .005)
    payload = {
        "training_version": TRAINING_VERSION, "strict_fold": 0,
        "count_weight": COUNT_WEIGHT, "baseline_macro_f1": baseline,
        "best_candidate_macro_f1": best, "delta": best - baseline,
        "epochs": epochs, "adopted_for_full_cv": adopted,
    }
    RESULTS_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    return payload


if __name__ == "__main__":
    train_fold0()
