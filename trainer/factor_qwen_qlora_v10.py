"""One-fold supervised Qwen2.5-3B QLoRA gate for Subtask 2 (V10)."""
from __future__ import annotations

import json
import math

import numpy as np
import torch
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          BitsAndBytesConfig, get_cosine_schedule_with_warmup)

from configs.config import config
from inference.factor_nli import _rank_decode
from preprocess.preprocess import load_train_data
from trainer.factor_llm_lexical_v6 import _current_v3_probability
from trainer.factor_train import WeightedGroupedASL
from utils.seed import seed_everything


OUTPUT = config.OUTPUT_DIR / "factor_qwen_qlora_v10"
RESULTS = OUTPUT / "fold0_results.json"
ADAPTER = OUTPUT / "fold0_adapter"
TRAINING_VERSION = "factor-qwen25-3b-supervised-qlora-v10"
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
MAX_LENGTH = 256
BATCH_SIZE = 2
ACCUMULATION = 8
EPOCHS = 1
FIXED_WEIGHT = .15
SEED = 101010
PREFIX = (
    "Detect every explicitly supported suicide-related risk and protective factor "
    "in this Reddit post. Multi-label research classification.\nPost:\n"
)


class FactorPromptDataset(Dataset):
    def __init__(self, texts, targets, counts, tokenizer):
        prefix = tokenizer.encode(PREFIX, add_special_tokens=False)
        suffix = tokenizer.encode("\nFactors:", add_special_tokens=False)
        available = MAX_LENGTH - len(prefix) - len(suffix) - 1
        first = available // 2
        self.rows = []
        for text, target, count in tqdm(
            zip(texts, targets, counts), total=len(texts), desc="V10 factor tokenization"
        ):
            post = tokenizer.encode(str(text), add_special_tokens=False)
            if len(post) > available:
                post = post[:first] + post[-(available - first):]
            ids = prefix + post + suffix + [tokenizer.eos_token_id]
            mask = [1] * len(ids); padding = MAX_LENGTH - len(ids)
            ids += [tokenizer.pad_token_id] * padding; mask += [0] * padding
            self.rows.append({
                "input_ids": torch.tensor(ids, dtype=torch.long),
                "attention_mask": torch.tensor(mask, dtype=torch.long),
                "targets": torch.tensor(target, dtype=torch.float32),
                "counts": torch.tensor(count, dtype=torch.float32),
            })

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def _model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    quantization = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=config.NUM_FACTORS,
        quantization_config=quantization, device_map={"": 0},
        local_files_only=True,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    model.config.problem_type = "multi_label_classification"
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.SEQ_CLS, r=8, lora_alpha=16, lora_dropout=.05,
        target_modules=["q_proj", "v_proj"], modules_to_save=["score"],
    ))
    return model, tokenizer


def _loader(dataset, indices, shuffle):
    return DataLoader(
        Subset(dataset, list(map(int, indices))), batch_size=BATCH_SIZE,
        shuffle=shuffle, num_workers=0, pin_memory=True,
    )


def _percentile(values):
    result = np.empty_like(values, dtype=np.float32); n = len(values)
    for label in range(values.shape[1]):
        order = np.argsort(values[:, label], kind="mergesort")
        result[order, label] = np.arange(n, dtype=np.float32) / max(1, n - 1)
    return result


@torch.no_grad()
def _infer(model, dataset, indices):
    model.eval(); rows = []
    for batch in tqdm(_loader(dataset, indices, False), desc="V10 untouched outer users"):
        with torch.autocast(device_type="cuda", enabled=True):
            logits = model(
                input_ids=batch["input_ids"].cuda(non_blocking=True),
                attention_mask=batch["attention_mask"].cuda(non_blocking=True),
            ).logits
        rows.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.vstack(rows)


def train_fold0():
    if not torch.cuda.is_available():
        raise RuntimeError("Factor Qwen V10 requires CUDA")
    OUTPUT.mkdir(parents=True, exist_ok=True); seed_everything(SEED)
    frame = load_train_data().reset_index(drop=True)
    targets = np.vstack(frame.factor_vector.to_numpy()).astype(np.float32)
    counts = np.vstack(frame.factor_counts.to_numpy()).astype(np.float32)
    risk = frame.risk_label.to_numpy(np.int64)
    groups = frame.anon_user_id.astype(str).to_numpy()
    train_idx, valid_idx = next(StratifiedGroupKFold(
        n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED,
    ).split(np.zeros(len(frame)), risk, groups))
    model, tokenizer = _model_and_tokenizer()
    dataset = FactorPromptDataset(frame.text.astype(str).tolist(), targets, counts, tokenizer)
    positives = torch.tensor(targets[train_idx].sum(0), device="cuda")
    weights = torch.sqrt((len(train_idx) - positives) / positives.clamp_min(1.)).clamp(1., 12.)
    loss_fn = WeightedGroupedASL(weights)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = AdamW(trainable, lr=1.5e-4, weight_decay=.01)
    loader = _loader(dataset, train_idx, True)
    updates = math.ceil(len(loader) / ACCUMULATION) * EPOCHS
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, max(1, int(.06 * updates)), max(1, updates)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    history = []
    for epoch in range(1, EPOCHS + 1):
        model.train(); optimizer.zero_grad(set_to_none=True); losses = []
        for step, batch in enumerate(tqdm(loader, desc=f"V10 QLoRA epoch {epoch}"), 1):
            target = batch["targets"].cuda(non_blocking=True)
            occurrence = batch["counts"].cuda(non_blocking=True)
            with torch.autocast(device_type="cuda", enabled=True):
                logits = model(
                    input_ids=batch["input_ids"].cuda(non_blocking=True),
                    attention_mask=batch["attention_mask"].cuda(non_blocking=True),
                ).logits.float()
                loss = loss_fn(logits, target, occurrence) / ACCUMULATION
            scaler.scale(loss).backward(); losses.append(float(loss.detach()) * ACCUMULATION)
            if step % ACCUMULATION == 0 or step == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, 1.)
                old_scale = scaler.get_scale(); scaler.step(optimizer); scaler.update()
                if scaler.get_scale() >= old_scale:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses))}
        history.append(row); print(json.dumps(row), flush=True)
    qwen = _infer(model, dataset, valid_idx)
    model.save_pretrained(ADAPTER); tokenizer.save_pretrained(ADAPTER)
    base, _ = _current_v3_probability(); base = base[valid_idx]
    prevalence = targets[train_idx].mean(0)
    truth = targets[valid_idx].astype(np.int8)
    baseline_prediction = _rank_decode(base, prevalence, 1.10)
    qwen_prediction = _rank_decode(qwen, prevalence, 1.10)
    candidate_probability = ((1. - FIXED_WEIGHT) * _percentile(base)
                             + FIXED_WEIGHT * _percentile(qwen))
    candidate_prediction = _rank_decode(candidate_probability, prevalence, 1.10)
    baseline = float(f1_score(truth, baseline_prediction, average="macro", zero_division=0))
    standalone = float(f1_score(truth, qwen_prediction, average="macro", zero_division=0))
    candidate = float(f1_score(truth, candidate_prediction, average="macro", zero_division=0))
    users = np.unique(groups[valid_idx]); rng = np.random.default_rng(SEED); deltas = []
    valid_groups = groups[valid_idx]
    for _ in range(3000):
        sampled = rng.choice(users, size=len(users), replace=True)
        positions = np.concatenate([np.flatnonzero(valid_groups == user) for user in sampled])
        old = f1_score(truth[positions], baseline_prediction[positions], average="macro", zero_division=0)
        new = f1_score(truth[positions], candidate_prediction[positions], average="macro", zero_division=0)
        deltas.append(float(new - old))
    deltas = np.asarray(deltas)
    bootstrap = {
        "mean_delta": float(deltas.mean()), "p05_delta": float(np.quantile(deltas, .05)),
        "p95_delta": float(np.quantile(deltas, .95)),
        "positive_fraction": float((deltas > 0).mean()),
    }
    promising = bool(candidate >= baseline + .005 and bootstrap["positive_fraction"] >= .80)
    payload = {
        "training_version": TRAINING_VERSION, "model": MODEL_NAME,
        "evaluation_scope": "one untouched outer user fold",
        "history": history, "baseline_macro_f1": baseline,
        "qwen_standalone_macro_f1": standalone,
        "fixed_blend_macro_f1": candidate, "fixed_weight": FIXED_WEIGHT,
        "changed_cells": int(np.sum(candidate_prediction != baseline_prediction)),
        "user_cluster_bootstrap": bootstrap,
        "promising_for_full_oof": promising, "adopted": False,
    }
    RESULTS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    return payload


if __name__ == "__main__":
    train_fold0()
