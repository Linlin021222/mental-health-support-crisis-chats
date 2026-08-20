"""Classic BERT baseline for both competition subtasks.

The baseline deliberately avoids the project's domain-adapted and ensemble
components.  One ``bert-base-uncased`` encoder feeds three simple heads:

* four-way suicide-risk classification;
* token-wise evidence classification; and
* 24 independent factor logits.

Evaluation uses the project's fixed five-fold user-disjoint membership and the
official containment/length Phrase-F1 implementation.  The script is intended
as a transparent thesis baseline, not as the production submission system.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baseline import _post_phrase_f1
from configs.config import config
from preprocess.preprocess import load_train_data
from utils.factor_calibration import apply_prior_topk
from utils.task1_metric import composite_score, task1_score


MODEL_NAME = "bert-base-uncased"
OUTPUT_DIR = config.OUTPUT_DIR / "bert_multitask_baseline"
SPLIT_FILE = PROJECT_ROOT / "kaggle_factor_qwen_package/factor_baseline_oof.npz"
SEED = 42


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _evidence_character_spans(text: str, phrases: list[str]):
    spans = []
    for phrase in phrases:
        phrase = str(phrase).strip()
        if not phrase:
            continue
        for match in re.finditer(re.escape(phrase), text, flags=re.IGNORECASE):
            spans.append((match.start(), match.end()))
    return spans


class BertDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, tokenizer, max_length: int):
        self.rows = []
        for row in tqdm(frame.itertuples(index=False), total=len(frame),
                        desc="Tokenising BERT baseline"):
            encoded = tokenizer(
                str(row.text), truncation=True, max_length=max_length,
                padding="max_length", return_offsets_mapping=True,
                return_tensors="pt",
            )
            offsets = encoded.pop("offset_mapping").squeeze(0).tolist()
            input_ids = encoded["input_ids"].squeeze(0)
            attention = encoded["attention_mask"].squeeze(0)
            evidence = torch.zeros(len(offsets), dtype=torch.float32)
            spans = _evidence_character_spans(str(row.text), list(row.evidence))
            for token, (start, end) in enumerate(offsets):
                if end > start and any(start < gold_end and end > gold_start
                                       for gold_start, gold_end in spans):
                    evidence[token] = 1.0
            valid_tokens = torch.tensor(
                [end > start for start, end in offsets], dtype=torch.bool
            ) & attention.bool()
            self.rows.append({
                "row_id": str(row.row_id),
                "user_id": str(row.anon_user_id),
                "text": str(row.text),
                "input_ids": input_ids,
                "attention_mask": attention,
                "valid_tokens": valid_tokens,
                "offsets": offsets,
                "risk": int(row.risk_label),
                "factor": torch.tensor(row.factor_vector, dtype=torch.float32),
                "evidence": evidence,
                "gold_evidence": list(row.evidence),
            })

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def collate(rows):
    return {
        "indices": torch.tensor([row["index"] for row in rows]),
        "input_ids": torch.stack([row["input_ids"] for row in rows]),
        "attention_mask": torch.stack([row["attention_mask"] for row in rows]),
        "valid_tokens": torch.stack([row["valid_tokens"] for row in rows]),
        "risk": torch.tensor([row["risk"] for row in rows]),
        "factor": torch.stack([row["factor"] for row in rows]),
        "evidence": torch.stack([row["evidence"] for row in rows]),
    }


class IndexedSubset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = list(map(int, indices))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        original = self.indices[index]
        row = dict(self.dataset[original])
        row["index"] = original
        return row


class BertMultiTaskBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(MODEL_NAME)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.risk_head = nn.Linear(hidden, 4)
        self.factor_head = nn.Linear(hidden, 24)
        self.evidence_head = nn.Linear(hidden, 1)

    def forward(self, input_ids, attention_mask):
        hidden = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        pooled = self.dropout(hidden[:, 0])
        return {
            "risk": self.risk_head(pooled),
            "factor": self.factor_head(pooled),
            "evidence": self.evidence_head(self.dropout(hidden)).squeeze(-1),
        }


def _weights(frame, indices, device):
    risks = frame.risk_label.to_numpy()[indices]
    risk_support = np.bincount(risks, minlength=4).astype(np.float32)
    risk_weight = np.sqrt(len(indices) / np.maximum(risk_support, 1.0))
    risk_weight /= risk_weight.mean()
    factors = np.vstack(frame.factor_vector.to_numpy())[indices]
    positive = factors.sum(0)
    factor_weight = np.sqrt(
        (len(indices) - positive) / np.maximum(positive, 1.0)
    ).clip(1.0, 10.0)
    return (
        torch.tensor(risk_weight, dtype=torch.float32, device=device),
        torch.tensor(factor_weight, dtype=torch.float32, device=device),
        factors.mean(0),
    )


def _decode_evidence(row, probability, predicted_risk,
                     threshold=0.45, max_tokens=12, topk=5):
    if int(predicted_risk) == config.RISK_LABELS["Indicator"]:
        return []
    offsets = row["offsets"]
    valid = row["valid_tokens"].numpy().astype(bool)
    active = (probability >= threshold) & valid
    candidates = []
    start = None
    for token in range(len(active) + 1):
        on = token < len(active) and active[token]
        if on and start is None:
            start = token
        if start is not None and (not on or token - start >= max_tokens):
            end_token = token - 1
            char_start, char_end = offsets[start][0], offsets[end_token][1]
            phrase = row["text"][char_start:char_end].strip()
            if phrase:
                score = float(np.mean(probability[start:end_token + 1]))
                candidates.append((score, phrase))
            start = token if on else None
    selected = []
    for _, phrase in sorted(candidates, reverse=True):
        normal = " ".join(phrase.casefold().split())
        if not any(normal in old.casefold() or old.casefold() in normal
                   for old in selected):
            selected.append(phrase)
        if len(selected) == topk:
            break
    return selected


@torch.no_grad()
def evaluate(model, loader, dataset, device, train_prevalence):
    model.eval()
    indices, risks, risk_probability = [], [], []
    factors, factor_probability, evidence_probability = [], [], []
    for batch in tqdm(loader, desc="BERT validation"):
        output = model(
            batch["input_ids"].to(device), batch["attention_mask"].to(device)
        )
        indices.extend(batch["indices"].tolist())
        risks.extend(batch["risk"].tolist())
        risk_probability.append(torch.softmax(output["risk"], -1).cpu().numpy())
        factors.append(batch["factor"].numpy())
        factor_probability.append(torch.sigmoid(output["factor"]).cpu().numpy())
        evidence_probability.append(torch.sigmoid(output["evidence"]).cpu().numpy())
    indices = np.asarray(indices)
    risks = np.asarray(risks)
    risk_probability = np.vstack(risk_probability)
    risk_prediction = risk_probability.argmax(1)
    factors = np.vstack(factors)
    factor_probability = np.vstack(factor_probability)
    factor_prediction = apply_prior_topk(
        factor_probability, train_prevalence, ratio=1.0
    )
    evidence_probability = np.vstack(evidence_probability)
    predicted_evidence = [
        _decode_evidence(dataset[int(index)], probability, risk)
        for index, probability, risk in zip(
            indices, evidence_probability, risk_prediction
        )
    ]
    gold_evidence = [dataset[int(index)]["gold_evidence"] for index in indices]
    phrase_scores = np.asarray([
        _post_phrase_f1(predicted, gold)
        for predicted, gold in zip(predicted_evidence, gold_evidence)
    ])
    risk_f1 = float(f1_score(risks, risk_prediction, average="weighted"))
    phrase_f1 = float(phrase_scores.mean())
    factor_f1 = float(f1_score(
        factors, factor_prediction, average="macro", zero_division=0
    ))
    metrics = {
        "risk_weighted_f1": risk_f1,
        "phrase_f1": phrase_f1,
        "task1": task1_score(risk_f1, phrase_f1),
        "task2_macro_f1": factor_f1,
        "composite": composite_score(risk_f1, phrase_f1, factor_f1),
        "risk_confusion": confusion_matrix(risks, risk_prediction).tolist(),
        "risk_report": classification_report(
            risks, risk_prediction,
            target_names=[config.ID2RISK[i] for i in range(4)],
            output_dict=True, zero_division=0,
        ),
    }
    predictions = pd.DataFrame({
        "row_id": [dataset[int(index)]["row_id"] for index in indices],
        "anon_user_id": [dataset[int(index)]["user_id"] for index in indices],
        "gold_risk": [config.ID2RISK[int(value)] for value in risks],
        "predicted_risk": [config.ID2RISK[int(value)] for value in risk_prediction],
        "gold_evidence": ["; ".join(value) for value in gold_evidence],
        "predicted_evidence": ["; ".join(value) for value in predicted_evidence],
        "phrase_f1": phrase_scores,
        "gold_factors": [str([config.ID2FACTOR[j] for j, x in enumerate(row) if x])
                         for row in factors],
        "predicted_factors": [str([config.ID2FACTOR[j] for j, x in enumerate(row) if x])
                              for row in factor_prediction],
    })
    return metrics, predictions


def run(args):
    seed_everything(SEED)
    frame = load_train_data().reset_index(drop=True)
    split = np.load(SPLIT_FILE, allow_pickle=True)
    if not np.array_equal(
        split["row_id"].astype(str), frame.row_id.astype(str).to_numpy()
    ):
        raise ValueError("BERT baseline split row order differs from train.xlsx")
    membership = split["fold_membership"].astype(int)
    train_indices = np.flatnonzero(membership != args.fold)
    valid_indices = np.flatnonzero(membership == args.fold)
    train_users = set(frame.anon_user_id.iloc[train_indices].astype(str))
    valid_users = set(frame.anon_user_id.iloc[valid_indices].astype(str))
    if train_users & valid_users:
        raise RuntimeError("User leakage in BERT baseline split")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    dataset = BertDataset(frame, tokenizer, args.max_length)
    train_loader = DataLoader(
        IndexedSubset(dataset, train_indices), batch_size=args.batch_size,
        shuffle=True, collate_fn=collate, num_workers=0, pin_memory=True,
    )
    valid_loader = DataLoader(
        IndexedSubset(dataset, valid_indices), batch_size=args.batch_size * 2,
        shuffle=False, collate_fn=collate, num_workers=0, pin_memory=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertMultiTaskBaseline().to(device)
    risk_weight, factor_weight, prevalence = _weights(
        frame, train_indices, device
    )
    risk_loss = nn.CrossEntropyLoss(weight=risk_weight)
    factor_loss = nn.BCEWithLogitsLoss(pos_weight=factor_weight)
    evidence_loss = nn.BCEWithLogitsLoss(reduction="none", pos_weight=torch.tensor(
        8.0, device=device
    ))
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    updates = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, max(1, int(0.1 * updates)), max(1, updates)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train(); losses = []
        progress = tqdm(train_loader, desc=f"BERT baseline epoch {epoch}")
        for batch in progress:
            ids = batch["input_ids"].to(device)
            attention = batch["attention_mask"].to(device)
            valid = batch["valid_tokens"].to(device)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                output = model(ids, attention)
                r_loss = risk_loss(output["risk"], batch["risk"].to(device))
                f_loss = factor_loss(output["factor"], batch["factor"].to(device))
                token_loss = evidence_loss(
                    output["evidence"], batch["evidence"].to(device)
                )
                e_loss = token_loss[valid].mean()
                loss = r_loss + 0.60 * f_loss + 0.80 * e_loss
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update(); scheduler.step()
            losses.append(float(loss.detach()))
            progress.set_postfix(loss=f"{np.mean(losses[-50:]):.4f}")
        metrics, _ = evaluate(model, valid_loader, dataset, device, prevalence)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), **metrics})
        print(json.dumps(history[-1], indent=2), flush=True)

    metrics, predictions = evaluate(
        model, valid_loader, dataset, device, prevalence
    )
    result = {
        "training_version": "classic-bert-multitask-baseline-v1",
        "model": MODEL_NAME,
        "fold": args.fold,
        "train_posts": int(len(train_indices)),
        "valid_posts": int(len(valid_indices)),
        "train_users": int(len(train_users)),
        "valid_users": int(len(valid_users)),
        "user_overlap": 0,
        "epochs": args.epochs,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "loss": "CE_risk + 0.60*weighted_BCE_factors + 0.80*token_BCE_evidence",
        "history": history,
        "final": metrics,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "fold0_results.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    predictions.to_csv(OUTPUT_DIR / "fold0_predictions.csv", index=False)
    if args.save_model:
        torch.save(model.state_dict(), OUTPUT_DIR / "fold0_model.pt")
    print(json.dumps(result, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, choices=range(5), default=0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--save-model", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
