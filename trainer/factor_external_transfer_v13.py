"""External mental-health transfer plus joint risk supervision for Task 2.

Stage A adapts MentalRoBERTa on the public IMHI MultiWD and IRF Reddit tasks.
Near-duplicate external posts are removed against *all* PFA posts before any
training. Stage B initialises the 24-factor model with that encoder and trains
on one strict, user-disjoint PFA fold with an auxiliary four-level risk head.

This is an isolated experiment: it never replaces production checkpoints.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.preprocessing import normalize
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from configs.config import config
from datasets.factor_cache_builder import build_factor_cache
from inference.factor_nli import _rank_decode
from models.factor_model import MentalRobertaFactorModel, factor_optimizer_parameters
from preprocess.preprocess import load_train_data
from trainer.factor_count_aux_v9 import _v3_with_replaced_semantic
from trainer.factor_llm_lexical_v6 import _current_v3_probability
from trainer.factor_train import FactorDataset, WeightedGroupedASL, _loader
from utils.seed import seed_everything


OUTPUT = config.OUTPUT_DIR / "factor_external_transfer_v13"
RESULTS = OUTPUT / "fold0_results.json"
EXTERNAL_ENCODER = OUTPUT / "external_encoder.pt"
BEST_MODEL = OUTPUT / "fold0_model.pt"
TRAINING_VERSION = "factor-imhi-transfer-joint-risk-v13"

EXTERNAL_ROOT = (config.DATA_DIR / "external" / "MentalLLaMA" / "train_data"
                 / "complete_data")
EXTERNAL_LABELS = (
    "thwarted belongingness", "perceived burdensomeness",
    "spiritual", "physical", "intellectual", "social", "vocational", "emotional",
)


def _clean_post(text):
    return re.sub(r"^\s*post:\s*", "", str(text), flags=re.I).strip()


def _normalise(text):
    return re.sub(r"\s+", " ", _clean_post(text).casefold()).strip()


def _external_rows(pfa_texts):
    """Aggregate question-per-row IMHI files into post-level masked labels."""
    if not EXTERNAL_ROOT.exists():
        raise FileNotFoundError(
            f"Missing {EXTERNAL_ROOT}. Clone https://github.com/SteveKGYang/MentalLLaMA "
            "to data/external/MentalLLaMA first."
        )
    aggregated = {}
    aliases = {
        "thwarted belongingness": 0, "perceived burdensomeness": 1,
        "spiritual": 2, "physical": 3, "intellectual": 4,
        "social": 5, "vocational": 6, "emotional": 7,
    }
    source_counts = defaultdict(int)
    for task in ("Irf", "MultiWD"):
        frames = [pd.read_csv(EXTERNAL_ROOT / task / f"{split}.csv")
                  for split in ("train", "val")]
        frame = pd.concat(frames, ignore_index=True)
        for row in frame.itertuples(index=False):
            text = _clean_post(row.post)
            key = _normalise(text)
            if key not in aggregated:
                aggregated[key] = {
                    "text": text, "targets": np.zeros(8, np.float32),
                    "mask": np.zeros(8, np.float32), "sources": set(),
                }
            question = str(row.question).casefold()
            matched = [name for name in aliases if name in question]
            if len(matched) != 1:
                continue
            label = aliases[matched[0]]
            answer = str(row.response).strip().casefold()
            target = float(answer.startswith("yes"))
            item = aggregated[key]
            item["targets"][label] = target
            item["mask"][label] = 1.0
            item["sources"].add(task)

    rows = list(aggregated.values())
    # Remove exact and high-similarity overlap with every PFA post.  Fitting
    # this unsupervised duplicate detector on all text is safe; no PFA label is
    # used and it prevents external-data leakage into the strict holdout.
    pfa_norm = [_normalise(x) for x in pfa_texts]
    ext_norm = [_normalise(x["text"]) for x in rows]
    exact = set(pfa_norm)
    keep = np.asarray([text not in exact for text in ext_norm])
    candidate_indices = np.flatnonzero(keep)
    all_text = pfa_norm + [ext_norm[i] for i in candidate_indices]
    vectorizer = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(4, 5), min_df=2,
        max_features=120000, dtype=np.float32,
    )
    matrix = vectorizer.fit_transform(all_text)
    pfa_matrix = normalize(matrix[:len(pfa_norm)])
    external_matrix = normalize(matrix[len(pfa_norm):])
    maximum = np.zeros(len(candidate_indices), dtype=np.float32)
    for start in range(0, len(candidate_indices), 256):
        stop = min(start + 256, len(candidate_indices))
        maximum[start:stop] = (external_matrix[start:stop] @ pfa_matrix.T).max(
            axis=1).toarray().ravel()
    near = set(candidate_indices[maximum >= 0.80].tolist())
    filtered = [row for index, row in enumerate(rows) if keep[index] and index not in near]
    for row in filtered:
        for source in row.pop("sources"):
            source_counts[source] += 1
    audit = {
        "raw_unique_posts": len(rows), "kept_posts": len(filtered),
        "exact_overlap_removed": int((~keep).sum()),
        "near_overlap_removed": len(near), "source_post_counts": dict(source_counts),
    }
    return filtered, audit


class ExternalDataset(Dataset):
    def __init__(self, rows, tokenizer):
        encoded = tokenizer(
            [row["text"] for row in rows], padding="max_length", truncation=True,
            max_length=256, return_tensors="pt",
        )
        self.ids = encoded["input_ids"]
        self.mask = encoded["attention_mask"]
        self.targets = torch.tensor(np.stack([x["targets"] for x in rows]))
        self.target_mask = torch.tensor(np.stack([x["mask"] for x in rows]))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        return self.ids[index], self.mask[index], self.targets[index], self.target_mask[index]


class ExternalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(config.FACTOR_MODEL_NAME, dtype=torch.float32)
        self.encoder.gradient_checkpointing_enable(); self.encoder.config.use_cache = False
        hidden = self.encoder.config.hidden_size
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Sequential(nn.Dropout(.15), nn.Linear(hidden, 8))

    def forward(self, ids, mask):
        hidden = self.encoder(input_ids=ids, attention_mask=mask).last_hidden_state.float()
        weight = mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * weight).sum(1) / weight.sum(1).clamp_min(1.0)
        return self.head(self.norm(pooled))


class JointRiskFactorModel(MentalRobertaFactorModel):
    def __init__(self):
        super().__init__()
        hidden = self.encoder.config.hidden_size
        self.risk_head = nn.Sequential(
            nn.Dropout(.15), nn.Linear(hidden, hidden // 2), nn.GELU(),
            nn.Dropout(.10), nn.Linear(hidden // 2, config.NUM_RISK_CLASSES),
        )

    def forward(self, input_ids, attention_mask):
        factor, semantic, pooled = super().forward(
            input_ids, attention_mask, return_semantic=True, return_features=True,
        )
        return factor, semantic, self.risk_head(pooled)


@torch.no_grad()
def _external_eval(model, loader, device):
    model.eval(); probabilities, targets, masks = [], [], []
    for ids, mask, target, target_mask in loader:
        logits = model(ids.to(device), mask.to(device))
        probabilities.append(torch.sigmoid(logits).cpu().numpy())
        targets.append(target.numpy()); masks.append(target_mask.numpy())
    probability = np.vstack(probabilities); target = np.vstack(targets); present = np.vstack(masks)
    scores = []
    for label in range(8):
        selected = present[:, label].astype(bool)
        scores.append(f1_score(target[selected, label], probability[selected, label] >= .5,
                               zero_division=0))
    return float(np.mean(scores)), scores


def _pretrain_external(rows, device):
    tokenizer = AutoTokenizer.from_pretrained(config.FACTOR_MODEL_NAME, use_fast=True)
    train_idx, valid_idx = train_test_split(
        np.arange(len(rows)), test_size=.10, random_state=config.SEED,
        stratify=np.asarray([int(x["mask"][:2].sum() > 0) for x in rows]),
    )
    dataset = ExternalDataset(rows, tokenizer)
    train_loader = DataLoader(torch.utils.data.Subset(dataset, train_idx), batch_size=8,
                              shuffle=True, num_workers=0, pin_memory=device.type == "cuda")
    valid_loader = DataLoader(torch.utils.data.Subset(dataset, valid_idx), batch_size=16,
                              shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    model = ExternalModel().to(device)
    parameters = [
        {"params": model.encoder.parameters(), "lr": 8e-6},
        {"params": list(model.norm.parameters()) + list(model.head.parameters()), "lr": 8e-5},
    ]
    optimizer = AdamW(parameters, weight_decay=config.WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    train_targets = dataset.targets[train_idx]
    train_masks = dataset.target_mask[train_idx]
    positives = (train_targets * train_masks).sum(0)
    negatives = ((1.0 - train_targets) * train_masks).sum(0)
    pos_weight = torch.sqrt(negatives / positives.clamp_min(1.0)).clamp(1., 8.).to(device)
    history = []
    for epoch in range(1, 3):
        model.train(); optimizer.zero_grad(set_to_none=True); losses = []
        for step, (ids, mask, targets, target_mask) in enumerate(
                tqdm(train_loader, desc=f"V13 external epoch {epoch}"), 1):
            ids=ids.to(device); mask=mask.to(device); targets=targets.to(device); target_mask=target_mask.to(device)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(ids, mask)
                raw = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, targets, pos_weight=pos_weight, reduction="none")
                loss = (raw * target_mask).sum() / target_mask.sum().clamp_min(1.0)
                loss = loss / 2
            scaler.scale(loss).backward(); losses.append(float(loss.detach()) * 2)
            if step % 2 == 0 or step == len(train_loader):
                scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
        macro, scores = _external_eval(model, valid_loader, device)
        row = {"epoch": epoch, "loss": float(np.mean(losses)),
               "external_macro_f1_at_0.5": macro, "per_label_f1": scores}
        history.append(row); print(json.dumps(row), flush=True)
    torch.save(model.encoder.state_dict(), EXTERNAL_ENCODER)
    del model
    if device.type == "cuda": torch.cuda.empty_cache()
    return history


@torch.no_grad()
def _factor_probability(model, loader, device):
    model.eval(); rows = []
    for batch in tqdm(loader, desc="V13 strict validation"):
        factor, _, _ = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        rows.append(torch.sigmoid(factor.float()).cpu().numpy())
    return np.vstack(rows)


def train_fold0():
    seed_everything(config.SEED + 1313); OUTPUT.mkdir(parents=True, exist_ok=True)
    frame = load_train_data().reset_index(drop=True)
    rows, audit = _external_rows(frame.text.astype(str).tolist())
    print(json.dumps({"external_audit": audit}), flush=True)
    device = torch.device(config.DEVICE)
    external_history = _pretrain_external(rows, device)

    cache = build_factor_cache(train=True); dataset = FactorDataset(cache)
    targets = np.vstack(frame.factor_vector.to_numpy()).astype(np.int8)
    risk = frame.risk_label.to_numpy(dtype=np.int64)
    groups = frame.anon_user_id.astype(str).to_numpy()
    fit_idx, valid_idx = next(StratifiedGroupKFold(
        n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED,
    ).split(np.zeros(len(frame)), risk, groups))
    prevalence = targets[fit_idx].mean(0)
    current, _ = _current_v3_probability()
    baseline_prediction = _rank_decode(current[valid_idx], prevalence, 1.10)
    baseline = float(f1_score(targets[valid_idx], baseline_prediction,
                              average="macro", zero_division=0))

    model = JointRiskFactorModel()
    missing, unexpected = model.encoder.load_state_dict(
        torch.load(EXTERNAL_ENCODER, map_location="cpu", weights_only=True), strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected external encoder keys: {unexpected}")
    model = model.to(device)
    factor_positive = torch.tensor(targets[fit_idx].sum(0), dtype=torch.float32)
    factor_weight = torch.sqrt((len(fit_idx) - factor_positive)
                               / factor_positive.clamp_min(1.)).clamp(1., 12.).to(device)
    factor_loss = WeightedGroupedASL(factor_weight).to(device)
    risk_counts = np.bincount(risk[fit_idx], minlength=config.NUM_RISK_CLASSES)
    risk_weight = torch.tensor(len(fit_idx) / np.maximum(risk_counts, 1),
                               dtype=torch.float32, device=device)
    risk_weight = risk_weight / risk_weight.mean()
    optimizer = AdamW(factor_optimizer_parameters(model), weight_decay=config.WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    train_loader = _loader(dataset, fit_idx, True)
    valid_loader = _loader(dataset, valid_idx, False)
    history=[]; best={"score": -1.0}; best_probability=None
    for epoch in range(1, config.FACTOR_EPOCHS + 1):
        model.train(); optimizer.zero_grad(set_to_none=True); losses=[]
        for step, batch in enumerate(tqdm(train_loader, desc=f"V13 PFA epoch {epoch}"), 1):
            ids=batch["input_ids"].to(device); mask=batch["attention_mask"].to(device)
            factor_targets=batch["factor_vectors"].to(device)
            counts=batch["factor_counts"].to(device)
            # Subset keeps original dataset indices in this loader's order only
            # through the batch itself, so risk labels are recovered by adding
            # them to the cached record collate below before training starts.
            risk_targets=batch["risk_labels"].to(device)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                factor_logits, semantic_logits, risk_logits=model(ids,mask)
                primary=factor_loss(factor_logits,factor_targets,counts)
                semantic=factor_loss(semantic_logits,factor_targets)
                auxiliary=torch.nn.functional.cross_entropy(risk_logits,risk_targets,
                                                             weight=risk_weight)
                loss=(primary + config.FACTOR_SEMANTIC_LOSS_WEIGHT*semantic + .15*auxiliary)
                loss=loss/config.GRADIENT_ACCUMULATION
            scaler.scale(loss).backward(); losses.append(float(loss.detach())*config.GRADIENT_ACCUMULATION)
            if step%config.GRADIENT_ACCUMULATION==0 or step==len(train_loader):
                scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
        probability=_factor_probability(model,valid_loader,device)
        semantic_pred=_rank_decode(probability,prevalence,1.10)
        standalone=float(f1_score(targets[valid_idx],semantic_pred,average="macro",zero_division=0))
        candidate_probability=_v3_with_replaced_semantic(probability,valid_idx)
        candidate_pred=_rank_decode(candidate_probability,prevalence,1.10)
        candidate=float(f1_score(targets[valid_idx],candidate_pred,average="macro",zero_division=0))
        row={"epoch":epoch,"train_loss":float(np.mean(losses)),
             "standalone_macro_f1":standalone,"candidate_macro_f1":candidate}
        history.append(row); print(json.dumps(row),flush=True)
        if candidate>best["score"]:
            best={"epoch":epoch,"score":candidate,"standalone":standalone}
            best_probability=probability.copy(); torch.save(model.state_dict(),BEST_MODEL)

    candidate_probability=_v3_with_replaced_semantic(best_probability,valid_idx)
    candidate_prediction=_rank_decode(candidate_probability,prevalence,1.10)
    per_label=[]
    for label,name in enumerate(config.FACTOR_LABELS):
        per_label.append({"label":name,"support":int(targets[valid_idx,label].sum()),
          "baseline_f1":float(f1_score(targets[valid_idx,label],baseline_prediction[:,label],zero_division=0)),
          "candidate_f1":float(f1_score(targets[valid_idx,label],candidate_prediction[:,label],zero_division=0))})
    payload={"training_version":TRAINING_VERSION,"strict_fold":0,"external_audit":audit,
             "external_history":external_history,"baseline_macro_f1":baseline,
             "best":best,"delta":best["score"]-baseline,"history":history,
             "per_label":per_label,"promising_for_full_oof":bool(best["score"]>=baseline+.005),
             "adopted":False}
    RESULTS.write_text(json.dumps(payload,indent=2),encoding="utf-8")
    print(json.dumps(payload,indent=2),flush=True); return payload


if __name__ == "__main__":
    train_fold0()
