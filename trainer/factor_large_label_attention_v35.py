"""DeBERTa-v3-large label-attention capacity gate for Task 2.

The 8 GB GPU cannot safely full-finetune the 435M backbone with Adam states.
We therefore keep the lower 18/24 layers frozen in FP16 and train the upper six
layers in FP32 plus a taxonomy-initialised label-specific attention head.  The
experiment has one fixed two-epoch fit and is evaluated once on untouched
outer-fold users; no outer-label model selection is performed.
"""
from __future__ import annotations

import json
import math

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup

from configs.config import config
from inference.factor_nli import _rank_decode
from preprocess.preprocess import load_train_data
from trainer.factor_llm_lexical_v6 import _current_v3_probability
from trainer.factor_sentence_evidence_cv_v27 import _user_bootstrap
from trainer.factor_train import WeightedGroupedASL
from utils.seed import seed_everything


OUTPUT = config.OUTPUT_DIR / "factor_large_label_attention_v35"
CHECKPOINT = OUTPUT / "fold0_trainable.pt"
PREDICTIONS = OUTPUT / "fold0_valid.npz"
RESULTS = OUTPUT / "fold0_results.json"
TRAINING_VERSION = "deberta-v3-large-label-attention-v35"
MODEL_NAME = "microsoft/deberta-v3-large"
MAX_LENGTH = 512
TOP_TRAINABLE_LAYERS = 6
EPOCHS = 2
ACCUMULATION = 8
BLEND_WEIGHT = 0.10
TOPK_RATIO = 1.10


class FactorDataset(Dataset):
    def __init__(self, texts, targets):
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME, use_fast=True, local_files_only=True
        )
        content = MAX_LENGTH - tokenizer.num_special_tokens_to_add(pair=False)
        first = content // 2
        self.rows = []
        for text, target in tqdm(
            zip(texts, targets), total=len(texts), desc="V35 head-tail tokenization"
        ):
            ids = tokenizer.encode(str(text), add_special_tokens=False)
            if len(ids) > content:
                ids = ids[:first] + ids[-(content - first):]
            ids = [tokenizer.cls_token_id, *ids, tokenizer.sep_token_id]
            mask = [1] * len(ids); padding = MAX_LENGTH - len(ids)
            ids.extend([tokenizer.pad_token_id] * padding); mask.extend([0] * padding)
            self.rows.append({
                "input_ids": torch.tensor(ids, dtype=torch.long),
                "attention_mask": torch.tensor(mask, dtype=torch.long),
                "targets": torch.tensor(target, dtype=torch.float32),
            })

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


class LargeLabelAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(
            MODEL_NAME, dtype=torch.float16, local_files_only=True
        )
        self.encoder.config.use_cache = False
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False
        for layer in self.encoder.encoder.layer[-TOP_TRAINABLE_LAYERS:]:
            layer.float()
            for parameter in layer.parameters():
                parameter.requires_grad = True
        hidden = int(self.encoder.config.hidden_size)
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME, use_fast=True, local_files_only=True
        )
        encoded = tokenizer(
            list(config.FACTOR_NLI_HYPOTHESES), padding=True, truncation=True,
            max_length=96, return_tensors="pt",
        )
        embedding = self.encoder.get_input_embeddings().weight.detach().float().cpu()
        special = set(tokenizer.all_special_ids); vectors = []
        for ids in encoded["input_ids"]:
            keep = torch.tensor([int(x) not in special for x in ids], dtype=torch.bool)
            vectors.append(embedding[ids[keep]].mean(0))
        initial = torch.stack(vectors)
        self.norm = nn.LayerNorm(hidden)
        self.label_queries = nn.Parameter(initial.clone())
        self.label_weights = nn.Parameter(nn.functional.normalize(initial, dim=-1))
        self.label_bias = nn.Parameter(torch.zeros(config.NUM_FACTORS))
        self.global_risk = nn.Linear(hidden, 19)
        self.global_protective = nn.Linear(hidden, 5)
        self.dropout = nn.Dropout(.15)

    def forward(self, input_ids, attention_mask):
        hidden = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state.float()
        tokens = self.norm(hidden)
        queries = nn.functional.normalize(self.label_queries, dim=-1)
        token_norm = nn.functional.normalize(tokens, dim=-1)
        scores = torch.einsum("bth,kh->btk", token_norm, queries) / .12
        scores = scores.masked_fill(~attention_mask.bool().unsqueeze(-1), -1e4)
        attention = torch.softmax(scores, dim=1)
        label_repr = torch.einsum("btk,bth->bkh", attention, tokens)
        local = (
            self.dropout(label_repr) * self.label_weights.unsqueeze(0)
        ).sum(-1) / math.sqrt(tokens.size(-1)) + self.label_bias
        mask = attention_mask.to(tokens.dtype).unsqueeze(-1)
        mean = (tokens * mask).sum(1) / mask.sum(1).clamp_min(1.)
        global_logits = torch.cat((
            self.global_risk(mean), self.global_protective(mean)
        ), dim=-1)
        return local + global_logits


def _loader(dataset, indices, shuffle):
    return DataLoader(
        Subset(dataset, list(map(int, indices))), batch_size=1, shuffle=shuffle,
        num_workers=0, pin_memory=True,
    )


def _loss(targets, indices, device):
    positive = targets[indices].sum(0)
    weight = np.sqrt((len(indices) - positive) / np.maximum(positive, 1.))
    return WeightedGroupedASL(
        torch.tensor(np.clip(weight, 1., 12.), dtype=torch.float32, device=device)
    )


def _train(model, dataset, indices, targets, device):
    loader = _loader(dataset, indices, True)
    trainable = [p for p in model.parameters() if p.requires_grad]
    head_ids = {id(p) for name, p in model.named_parameters()
                if not name.startswith("encoder.")}
    optimizer = AdamW([
        {"params": [p for p in trainable if id(p) not in head_ids], "lr": 4e-6},
        {"params": [p for p in trainable if id(p) in head_ids], "lr": 3e-5},
    ], weight_decay=.01)
    updates = math.ceil(len(loader) / ACCUMULATION) * EPOCHS
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, max(1, int(updates * .08)), max(1, updates)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    loss_fn = _loss(targets, indices, device)
    history = []
    for epoch in range(1, EPOCHS + 1):
        model.train(); optimizer.zero_grad(set_to_none=True); losses = []
        for step, batch in enumerate(
            tqdm(loader, desc=f"V35 large factor epoch {epoch}/{EPOCHS}"), 1
        ):
            y = batch["targets"].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", enabled=True):
                logits = model(
                    batch["input_ids"].to(device, non_blocking=True),
                    batch["attention_mask"].to(device, non_blocking=True),
                )
                loss = loss_fn(logits, y) / ACCUMULATION
            scaler.scale(loss).backward(); losses.append(float(loss.detach()) * ACCUMULATION)
            if step % ACCUMULATION == 0 or step == len(loader):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(trainable, 1.0)
                old_scale = scaler.get_scale(); scaler.step(optimizer); scaler.update()
                if scaler.get_scale() >= old_scale:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses))}
        history.append(row); print(f"V35 epoch={epoch} loss={row['train_loss']:.6f}", flush=True)
    return history


@torch.no_grad()
def _predict(model, dataset, indices, device):
    model.eval(); rows = []
    for batch in tqdm(_loader(dataset, indices, False), desc="V35 validation"):
        with torch.autocast(device_type="cuda", enabled=True):
            logits = model(
                batch["input_ids"].to(device, non_blocking=True),
                batch["attention_mask"].to(device, non_blocking=True),
            )
        rows.append(torch.sigmoid(logits.float()).cpu().numpy()[0])
    return np.vstack(rows).astype(np.float32)


def _rank_columns(probability):
    result = np.zeros_like(probability, dtype=np.float32)
    for label in range(probability.shape[1]):
        order = np.argsort(probability[:, label], kind="mergesort")
        result[order, label] = np.linspace(0., 1., len(probability), dtype=np.float32)
    return result


def _macro_auc(truth, probability):
    values = [
        roc_auc_score(truth[:, label], probability[:, label])
        for label in range(config.NUM_FACTORS)
        if np.unique(truth[:, label]).size == 2
    ]
    return float(np.mean(values))


def train_fold0():
    if not torch.cuda.is_available():
        raise RuntimeError("V35 requires CUDA")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    seed_everything(config.SEED + 3500)
    frame = load_train_data().reset_index(drop=True)
    targets = np.vstack(frame.factor_vector.to_numpy()).astype(np.float32)
    risk = frame.risk_label.to_numpy(dtype=np.int64)
    groups = frame.anon_user_id.astype(str).to_numpy()
    train_idx, valid_idx = next(StratifiedGroupKFold(
        n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED,
    ).split(np.zeros(len(frame)), risk, groups))
    dataset = FactorDataset(frame.text.astype(str).tolist(), targets)
    if PREDICTIONS.exists():
        saved = np.load(PREDICTIONS)
        if (str(saved["training_version"]) == TRAINING_VERSION
                and np.array_equal(saved["valid_indices"], valid_idx)):
            probability = saved["probabilities"].astype(np.float32)
            history = json.loads(str(saved["history"]))
            print("V35 fold0 resumed", flush=True)
        else:
            raise RuntimeError("Stale V35 cache")
    else:
        device = torch.device("cuda")
        model = LargeLabelAttention().to(device)
        history = _train(model, dataset, train_idx, targets, device)
        probability = _predict(model, dataset, valid_idx, device)
        state = {
            name: value.detach().cpu() for name, value in model.state_dict().items()
            if not name.startswith("encoder.")
            or (name.startswith("encoder.encoder.layer.")
                and int(name.split(".")[3]) >= 24 - TOP_TRAINABLE_LAYERS)
        }
        torch.save({"training_version": TRAINING_VERSION, "state_dict": state}, CHECKPOINT)
        np.savez_compressed(
            PREDICTIONS, probabilities=probability, valid_indices=valid_idx,
            history=json.dumps(history), training_version=TRAINING_VERSION,
        )
        del model
        torch.cuda.empty_cache()

    current, _ = _current_v3_probability(); base = current[valid_idx]
    mixed = ((1. - BLEND_WEIGHT) * _rank_columns(base)
             + BLEND_WEIGHT * _rank_columns(probability))
    prevalence = targets[train_idx].mean(0); truth = targets[valid_idx].astype(np.int8)
    baseline_prediction = _rank_decode(base, prevalence, TOPK_RATIO)
    standalone_prediction = _rank_decode(probability, prevalence, TOPK_RATIO)
    candidate_prediction = _rank_decode(mixed, prevalence, TOPK_RATIO)
    baseline = float(f1_score(truth, baseline_prediction, average="macro", zero_division=0))
    standalone = float(f1_score(truth, standalone_prediction, average="macro", zero_division=0))
    candidate = float(f1_score(truth, candidate_prediction, average="macro", zero_division=0))
    bootstrap = _user_bootstrap(
        truth, baseline_prediction, candidate_prediction, groups[valid_idx],
        seed=353535, draws=3000,
    )
    per_label = [{
        "label": config.ID2FACTOR[label], "support": int(truth[:, label].sum()),
        "baseline_f1": float(f1_score(
            truth[:, label], baseline_prediction[:, label], zero_division=0
        )),
        "candidate_f1": float(f1_score(
            truth[:, label], candidate_prediction[:, label], zero_division=0
        )),
    } for label in range(config.NUM_FACTORS)]
    payload = {
        "training_version": TRAINING_VERSION,
        "evaluation": "fixed two-epoch fit; untouched outer user fold0",
        "model": MODEL_NAME,
        "method": {
            "max_length": MAX_LENGTH, "head_tail": True,
            "trainable_top_layers": TOP_TRAINABLE_LAYERS,
            "taxonomy_initialised_label_attention": True,
            "loss": "positive-weighted grouped asymmetric loss",
            "fixed_blend_weight": BLEND_WEIGHT, "topk_ratio": TOPK_RATIO,
        },
        "history": history,
        "large_standalone_macro_f1": standalone,
        "large_standalone_macro_auc": _macro_auc(truth, probability),
        "baseline_macro_f1": baseline, "candidate_macro_f1": candidate,
        "delta": candidate - baseline, "user_cluster_bootstrap": bootstrap,
        "per_label": per_label,
        "promising_for_full_oof": bool(
            candidate >= baseline + .005 and bootstrap["positive_fraction"] >= .70
        ),
        "adopted": False,
    }
    RESULTS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    return payload


if __name__ == "__main__":
    train_fold0()
