"""Supervised neural Task-1 comparison: vanilla DeBERTa vs TAPT initialisation."""
from __future__ import annotations

import json
import math

import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup

from configs.config import config
from preprocess.preprocess import load_train_data
from trainer.task1_atomic_v25 import _load_records
from trainer.task1_tapt_v73 import (
    CHECKPOINT as TAPT_CHECKPOINT, HeadTailDataset, INNER_CAL_FOLD,
    _evidence, _metric, _override, _polarity,
)
from utils.seed import seed_everything


OUTPUT = config.OUTPUT_DIR / "task1_tapt_supervised_v74"
RESULTS = OUTPUT / "results.json"
TRAINING_VERSION = "task1-supervised-neural-tapt-comparison-v74"
MAX_EPOCHS = 2
TOP_LAYERS = 4
ACCUMULATION = 8


class NeuralRiskModel(nn.Module):
    def __init__(self, tapt: bool):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(
            config.MODEL_NAME, local_files_only=True, dtype=torch.float32)
        if tapt:
            saved = torch.load(TAPT_CHECKPOINT, map_location="cpu", weights_only=False)
            adapted = {key[len("deberta."):]: value
                       for key, value in saved["state_dict"].items()
                       if key.startswith("deberta.")}
            missing, unexpected = self.encoder.load_state_dict(adapted, strict=False)
            if unexpected:
                raise RuntimeError(f"Unexpected TAPT encoder keys: {unexpected}")
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False
        for layer in self.encoder.encoder.layer[-TOP_LAYERS:]:
            for parameter in layer.parameters():
                parameter.requires_grad = True
        hidden = int(self.encoder.config.hidden_size)
        self.projection = nn.Sequential(
            nn.LayerNorm(hidden * 3), nn.Linear(hidden * 3, hidden),
            nn.GELU(), nn.Dropout(.15))
        self.classifier = nn.Linear(hidden, 4)
        self.ordinal = nn.Linear(hidden, 3)

    def forward(self, input_ids, attention_mask):
        hidden = self.encoder(input_ids=input_ids,
                              attention_mask=attention_mask).last_hidden_state
        weights = attention_mask.to(hidden.dtype).unsqueeze(-1)
        mean = (hidden * weights).sum(1) / weights.sum(1).clamp_min(1.0)
        maximum = hidden.masked_fill(~attention_mask.bool().unsqueeze(-1), -1e4).max(1).values
        document = self.projection(torch.cat((hidden[:, 0], mean, maximum), 1).float())
        return self.classifier(document), self.ordinal(document)


def _loader(dataset, indices, shuffle):
    return DataLoader(Subset(dataset, list(map(int, indices))), batch_size=1,
                      shuffle=shuffle, num_workers=0, pin_memory=True)


def _probability(logits, ordinal_logits):
    categorical = torch.softmax(logits.float(), -1)
    cumulative = torch.sigmoid(ordinal_logits.float())
    cumulative = torch.cummin(cumulative, dim=1).values
    ordinal = torch.stack((
        1 - cumulative[:, 0], cumulative[:, 0] - cumulative[:, 1],
        cumulative[:, 1] - cumulative[:, 2], cumulative[:, 2]), 1).clamp_min(1e-7)
    ordinal /= ordinal.sum(1, keepdim=True)
    return (.8 * categorical + .2 * ordinal).cpu().numpy()


@torch.no_grad()
def _infer(model, dataset, indices, device, description):
    model.eval(); rows = []
    for batch in tqdm(_loader(dataset, indices, False), desc=description):
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits, ordinal = model(batch["input_ids"].to(device, non_blocking=True),
                                    batch["attention_mask"].to(device, non_blocking=True))
        rows.append(_probability(logits, ordinal)[0])
    return np.vstack(rows)


def _weights(labels, indices, device):
    selected = labels[np.asarray(indices)]
    counts = np.bincount(selected, minlength=4).astype(float)
    class_weights = np.sqrt(len(selected) / np.maximum(counts, 1.0))
    class_weights /= class_weights.mean()
    ordinal = selected[:, None] > np.arange(3)[None, :]
    positive = ordinal.sum(0).astype(float)
    ordinal_weights = np.sqrt((len(selected) - positive) / np.maximum(positive, 1.0))
    return (torch.tensor(class_weights, dtype=torch.float32, device=device),
            torch.tensor(ordinal_weights, dtype=torch.float32, device=device))


class LabelledSubset(torch.utils.data.Dataset):
    def __init__(self, dataset, indices, labels):
        self.dataset = dataset; self.indices = list(map(int, indices)); self.labels = labels
    def __len__(self): return len(self.indices)
    def __getitem__(self, position):
        index = self.indices[position]; row = dict(self.dataset[index])
        row["risk_labels"] = torch.tensor(int(self.labels[index]), dtype=torch.long)
        return row


def _train_labelled(model, dataset, indices, labels, device, epochs, callback=None):
    loader = DataLoader(LabelledSubset(dataset, indices, labels), batch_size=1,
                        shuffle=True, num_workers=0, pin_memory=True)
    trainable = [p for p in model.parameters() if p.requires_grad]
    head_ids = {id(p) for module in (model.projection, model.classifier, model.ordinal)
                for p in module.parameters()}
    optimizer = AdamW([
        {"params": [p for p in trainable if id(p) not in head_ids], "lr": 7e-6},
        {"params": [p for p in trainable if id(p) in head_ids], "lr": 4e-5},
    ], weight_decay=.01)
    updates = math.ceil(len(loader) / ACCUMULATION) * epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, max(1, int(.08 * updates)), max(1, updates))
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    class_weights, ordinal_weights = _weights(labels, indices, device)
    history = []
    for epoch in range(1, epochs + 1):
        model.train(); optimizer.zero_grad(set_to_none=True); losses = []
        progress = tqdm(loader, desc=f"V74 supervised epoch {epoch}/{epochs}")
        for step, batch in enumerate(progress, 1):
            target = batch["risk_labels"].to(device, non_blocking=True)
            ordinal_target = (target[:, None] > torch.arange(3, device=device)[None, :]).float()
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits, ordinal = model(batch["input_ids"].to(device, non_blocking=True),
                                        batch["attention_mask"].to(device, non_blocking=True))
                ce = nn.functional.cross_entropy(logits.float(), target,
                                                 weight=class_weights,
                                                 label_smoothing=.03)
                ordinal_loss = nn.functional.binary_cross_entropy_with_logits(
                    ordinal.float(), ordinal_target, pos_weight=ordinal_weights)
                loss = (ce + .30 * ordinal_loss) / ACCUMULATION
            scaler.scale(loss).backward(); losses.append(float(loss.detach()) * ACCUMULATION)
            if step % ACCUMULATION == 0 or step == len(loader):
                scaler.unscale_(optimizer); nn.utils.clip_grad_norm_(trainable, 1.0)
                old_scale = scaler.get_scale(); scaler.step(optimizer); scaler.update()
                if scaler.get_scale() >= old_scale: scheduler.step()
                optimizer.zero_grad(set_to_none=True)
        row = {"epoch": epoch, "loss": float(np.mean(losses))}; history.append(row)
        print(f"V74 epoch={epoch} loss={row['loss']:.5f}", flush=True)
        if callback is not None: callback(epoch, model, row)
    return history


def _select(probability, base, records, truth, decoder):
    rows = []
    standalone = float(f1_score(truth, probability.argmax(1),
                                average="weighted", zero_division=0))
    for threshold in (.45, .55, .65, .75, .85, .95):
        prediction = _override(base, probability, threshold)
        evidence = [_evidence(row, risk, decoder) for row, risk in zip(records, prediction)]
        prediction, evidence = _polarity(records, prediction, evidence)
        score = _metric(truth, prediction, evidence, records)
        rows.append({"threshold": threshold, "standalone_risk_f1": standalone,
                     "changed": int((prediction != base).sum()), **score})
    return max(rows, key=lambda row: (row["task1"], row["risk_f1"]))


def main():
    if not torch.cuda.is_available(): raise RuntimeError("V74 requires CUDA")
    OUTPUT.mkdir(parents=True, exist_ok=True); seed_everything(config.SEED + 7474)
    frame = load_train_data().reset_index(drop=True)
    labels = frame.risk_label.to_numpy(dtype=np.int64)
    groups = frame.anon_user_id.astype(str).to_numpy()
    outer_train, outer_valid = next(StratifiedGroupKFold(
        n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED
    ).split(np.zeros(len(frame)), labels, groups))
    nested, membership_map, outer_raw = _load_records()
    inner_fit = np.asarray([i for i in outer_train if membership_map[int(i)] != INNER_CAL_FOLD])
    inner_cal = np.asarray([i for i in outer_train if membership_map[int(i)] == INNER_CAL_FOLD])
    nested_by_index = {int(row["global_index"]): row for row in nested}
    inner_records = [nested_by_index[int(i)] for i in inner_cal]
    outer_records = outer_raw["records"]
    decoder = json.loads((config.OUTPUT_DIR / "task1_evidence_v4" / "calibration.json")
                         .read_text(encoding="utf-8"))
    inner_base = np.asarray([int(row["risk"]) for row in inner_records])
    inner_base_evidence = [_evidence(row, risk, decoder) for row, risk in zip(inner_records, inner_base)]
    inner_base, inner_base_evidence = _polarity(inner_records, inner_base, inner_base_evidence)
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME, local_files_only=True,
                                               use_fast=True, fix_mistral_regex=True)
    dataset = HeadTailDataset(frame.text.astype(str).tolist(), tokenizer)
    device = torch.device("cuda"); experiments = []
    for name, use_tapt in (("control", False), ("tapt", True)):
        seed_everything(config.SEED + 7474)
        model = NeuralRiskModel(use_tapt).to(device); best = None
        def callback(epoch, current, row):
            nonlocal best
            probability = _infer(current, dataset, inner_cal, device,
                                 f"V74 {name} inner epoch {epoch}")
            selected = _select(probability, inner_base, inner_records,
                               labels[inner_cal], decoder)
            selected.update({"epoch": epoch, "initialization": name})
            row.update(selected)
            print(f"V74 {name} epoch={epoch} standalone={selected['standalone_risk_f1']:.6f} "
                  f"task1={selected['task1']:.6f} threshold={selected['threshold']}", flush=True)
            if best is None or (selected["task1"], selected["risk_f1"]) > (best["task1"], best["risk_f1"]):
                best = selected
        history = _train_labelled(model, dataset, inner_fit, labels, device, MAX_EPOCHS, callback)
        experiments.append({"initialization": name, "best": best, "history": history})
        del model; torch.cuda.empty_cache()
    selected = max((row["best"] for row in experiments),
                   key=lambda row: (row["task1"], row["risk_f1"]))
    print("V74 selected", json.dumps(selected, indent=2), flush=True)

    seed_everything(config.SEED + 7475)
    final = NeuralRiskModel(selected["initialization"] == "tapt").to(device)
    refit = _train_labelled(final, dataset, outer_train, labels, device,
                            int(selected["epoch"]))
    probability = _infer(final, dataset, outer_valid, device, "V74 untouched outer users")
    base = np.asarray([int(row["risk"]) for row in outer_records])
    base_evidence = [_evidence(row, risk, decoder) for row, risk in zip(outer_records, base)]
    base, base_evidence = _polarity(outer_records, base, base_evidence)
    candidate = _override(base, probability, selected["threshold"])
    candidate_evidence = [_evidence(row, risk, decoder) for row, risk in zip(outer_records, candidate)]
    candidate, candidate_evidence = _polarity(outer_records, candidate, candidate_evidence)
    baseline = _metric(labels[outer_valid], base, base_evidence, outer_records)
    result = _metric(labels[outer_valid], candidate, candidate_evidence, outer_records)
    unique = np.unique(groups[outer_valid]); rng = np.random.default_rng(config.SEED + 7474)
    deltas = []
    for _ in range(4000):
        users = rng.choice(unique, len(unique), replace=True)
        positions = np.concatenate([np.flatnonzero(groups[outer_valid] == user) for user in users])
        old = _metric(labels[outer_valid][positions], base[positions],
                      [base_evidence[i] for i in positions], [outer_records[i] for i in positions])["task1"]
        new = _metric(labels[outer_valid][positions], candidate[positions],
                      [candidate_evidence[i] for i in positions], [outer_records[i] for i in positions])["task1"]
        deltas.append(new - old)
    deltas = np.asarray(deltas)
    bootstrap = {"mean_delta": float(deltas.mean()), "p05_delta": float(np.quantile(deltas,.05)),
                 "p95_delta": float(np.quantile(deltas,.95)),
                 "positive_fraction": float((deltas>0).mean())}
    adopted = bool(selected["initialization"] == "tapt" and result["task1"] >= baseline["task1"]+.002
                   and bootstrap["positive_fraction"] >= .80 and bootstrap["p05_delta"] >= 0)
    payload = {"training_version": TRAINING_VERSION,
               "evaluation": "matched supervised neural models; untouched outer user holdout",
               "experiments": experiments, "selected": selected, "refit": refit,
               "baseline": baseline,
               "candidate": {**result, "changed": int((candidate != base).sum()),
                             "standalone_risk_f1": float(f1_score(labels[outer_valid], probability.argmax(1),
                                                                  average='weighted', zero_division=0))},
               "bootstrap": bootstrap, "adopted": adopted}
    RESULTS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True); return payload


if __name__ == "__main__": main()
