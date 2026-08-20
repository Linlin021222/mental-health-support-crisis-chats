"""Paper-aligned independent risk/protective representation continuation.

Li et al. (2025) explicitly model the 19 risk and five protective factors in
separate representation branches because protective factors are an independent
dimension, not the inverse of risk factors.  V36 adds that missing inductive
bias to the accepted MentalRoBERTa fold without modifying its parameters.
Small zero-initialised residual gates make epoch zero exactly reproduce the
accepted semantic component.
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
from tqdm import tqdm
from transformers import AutoTokenizer

from configs.config import config
from inference.factor_nli import _rank_decode
from models.factor_model import MentalRobertaFactorModel
from preprocess.factor_paper_definitions_v36 import PAPER_FACTOR_DEFINITIONS
from preprocess.preprocess import load_train_data
from trainer.factor_llm_lexical_v6 import _current_v3_probability
from trainer.factor_sentence_evidence_cv_v27 import _user_bootstrap
from trainer.factor_train import FactorDataset, WeightedGroupedASL, _loader
from utils.seed import seed_everything


OUTPUT = config.OUTPUT_DIR / "factor_paper_dual_branch_v36"
CHECKPOINT = OUTPUT / "fold0_residual.pt"
PREDICTIONS = OUTPUT / "fold0_valid.npz"
RESULTS = OUTPUT / "fold0_results.json"
TRAINING_VERSION = "paper-aligned-risk-protective-dual-branch-v36"
SOURCE = config.OUTPUT_DIR / "factor_cv" / "fold0_model.pt"
BASE_OOF = config.OUTPUT_DIR / "factor_cv" / "factor_oof_predictions.npz"
EPOCHS = 4
LEARNING_RATE = 2e-4
REPLACEMENT_WEIGHT = 0.30
TOPK_RATIO = 1.10


class PaperDualBranchModel(MentalRobertaFactorModel):
    def __init__(self):
        super().__init__(initialise_labels=False)
        hidden = int(self.encoder.config.hidden_size)
        tokenizer = AutoTokenizer.from_pretrained(
            config.FACTOR_MODEL_NAME, use_fast=True, local_files_only=True
        )
        encoded = tokenizer(
            PAPER_FACTOR_DEFINITIONS, padding=True, truncation=True,
            max_length=80, return_tensors="pt",
        )
        was_training = self.encoder.training; self.encoder.eval()
        with torch.no_grad():
            definition_hidden = self.encoder(**encoded).last_hidden_state.float()
        if was_training:
            self.encoder.train()
        mask = encoded["attention_mask"].unsqueeze(-1).float()
        paper = (definition_hidden * mask).sum(1) / mask.sum(1).clamp_min(1.)
        self.paper_queries = nn.Parameter(paper)

        # Independent residual subspaces following Eq. (6)-(7) of the paper.
        self.risk_branch = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Dropout(.10), nn.Linear(hidden // 2, hidden), nn.ReLU(),
        )
        self.protective_branch = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Dropout(.10), nn.Linear(hidden // 2, hidden), nn.ReLU(),
        )
        self.risk_residual = nn.Linear(hidden, 19)
        self.protective_residual = nn.Linear(hidden, 5)
        nn.init.zeros_(self.risk_residual.weight); nn.init.zeros_(self.risk_residual.bias)
        nn.init.zeros_(self.protective_residual.weight); nn.init.zeros_(self.protective_residual.bias)
        self.definition_gate = nn.Parameter(torch.tensor(-2.2))
        self.risk_gate = nn.Parameter(torch.tensor(-2.2))
        self.protective_gate = nn.Parameter(torch.tensor(-2.2))

    def forward(self, input_ids, attention_mask, return_residual=False):
        batch, chunks, length = input_ids.shape
        ids = input_ids.reshape(batch * chunks, length)
        flat_mask = attention_mask.reshape(batch * chunks, length)
        with torch.no_grad():
            hidden = self.encoder(input_ids=ids, attention_mask=flat_mask).last_hidden_state
        tokens = self.norm(hidden.float().reshape(batch, chunks * length, -1))
        mask = attention_mask.reshape(batch, chunks * length).bool()
        definition_mix = torch.sigmoid(self.definition_gate)
        queries = self.label_queries + definition_mix * self.paper_queries
        scores = torch.einsum("bth,kh->btk", tokens, queries) / math.sqrt(tokens.size(-1))
        scores = scores.masked_fill(~mask.unsqueeze(-1), -1e4)
        attention = torch.softmax(scores, dim=1)
        label_repr = torch.einsum("btk,bth->bkh", attention, tokens)
        local = (
            self.dropout(label_repr) * self.label_weights.unsqueeze(0)
        ).sum(-1) + self.label_bias
        float_mask = mask.unsqueeze(-1).to(tokens.dtype)
        global_repr = (tokens * float_mask).sum(1) / float_mask.sum(1).clamp_min(1.)
        accepted_global = torch.cat((
            self.global_risk(global_repr), self.global_protective(global_repr)
        ), dim=-1)
        risk = self.risk_residual(self.risk_branch(global_repr))
        protective = self.protective_residual(self.protective_branch(global_repr))
        residual = torch.cat((
            torch.sigmoid(self.risk_gate) * risk,
            torch.sigmoid(self.protective_gate) * protective,
        ), dim=-1)
        logits = local + accepted_global + residual
        return (logits, residual) if return_residual else logits


def _freeze_accepted(model):
    for parameter in model.parameters():
        parameter.requires_grad = False
    prefixes = (
        "paper_queries", "risk_branch.", "protective_branch.",
        "risk_residual.", "protective_residual.",
        "definition_gate", "risk_gate", "protective_gate",
    )
    for name, parameter in model.named_parameters():
        if name.startswith(prefixes):
            parameter.requires_grad = True


def _loss(targets, train_idx, device):
    positive = targets[train_idx].sum(0)
    weights = np.sqrt((len(train_idx) - positive) / np.maximum(positive, 1.))
    return WeightedGroupedASL(torch.tensor(
        np.clip(weights, 1., 12.), dtype=torch.float32, device=device
    ))


def _train(model, dataset, train_idx, targets, device):
    loader = _loader(dataset, train_idx, True)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable, lr=LEARNING_RATE, weight_decay=.01)
    scaler = torch.amp.GradScaler("cuda", enabled=config.FP16)
    loss_fn = _loss(targets, train_idx, device)
    history = []
    for epoch in range(1, EPOCHS + 1):
        model.train(); optimizer.zero_grad(set_to_none=True); losses = []
        for batch in tqdm(loader, desc=f"V36 dual branch epoch {epoch}/{EPOCHS}"):
            y = batch["factor_vectors"].to(device)
            with torch.autocast(device_type="cuda", enabled=config.FP16):
                logits = model(
                    batch["input_ids"].to(device), batch["attention_mask"].to(device)
                )
                loss = loss_fn(logits, y)
            scaler.scale(loss).backward(); losses.append(float(loss.detach()))
            scaler.unscale_(optimizer); nn.utils.clip_grad_norm_(trainable, 1.)
            scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses))})
        print(f"V36 epoch={epoch} loss={history[-1]['train_loss']:.6f}", flush=True)
    return history


@torch.no_grad()
def _predict(model, dataset, indices, device):
    model.eval(); probabilities = []
    for batch in tqdm(_loader(dataset, indices, False), desc="V36 validation"):
        with torch.autocast(device_type="cuda", enabled=config.FP16):
            logits = model(
                batch["input_ids"].to(device), batch["attention_mask"].to(device)
            )
        probabilities.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.vstack(probabilities).astype(np.float32)


def _macro_auc(truth, probability):
    return float(np.mean([
        roc_auc_score(truth[:, label], probability[:, label])
        for label in range(config.NUM_FACTORS)
        if np.unique(truth[:, label]).size == 2
    ]))


def train_fold0():
    if not torch.cuda.is_available():
        raise RuntimeError("V36 requires CUDA")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    seed_everything(config.SEED + 3600)
    frame = load_train_data().reset_index(drop=True)
    targets = np.vstack(frame.factor_vector.to_numpy()).astype(np.float32)
    risk = frame.risk_label.to_numpy(dtype=np.int64)
    groups = frame.anon_user_id.astype(str).to_numpy()
    train_idx, valid_idx = next(StratifiedGroupKFold(
        n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED
    ).split(np.zeros(len(frame)), risk, groups))
    dataset = FactorDataset(config.CACHE_DIR / "factor_train_cache.pt")
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)
    if PREDICTIONS.exists():
        saved = np.load(PREDICTIONS)
        if (str(saved["training_version"]) == TRAINING_VERSION
                and np.array_equal(saved["valid_indices"], valid_idx)):
            probability = saved["probabilities"].astype(np.float32)
            history = json.loads(str(saved["history"])); print("V36 resumed", flush=True)
        else:
            raise RuntimeError("Stale V36 cache")
    else:
        device = torch.device("cuda")
        model = PaperDualBranchModel().to(device)
        model.load_state_dict(torch.load(SOURCE, map_location="cpu", weights_only=True), strict=False)
        _freeze_accepted(model)
        print(
            f"V36 trainable parameters={sum(p.numel() for p in model.parameters() if p.requires_grad):,}",
            flush=True,
        )
        history = _train(model, dataset, train_idx, targets, device)
        probability = _predict(model, dataset, valid_idx, device)
        state = {name: value.detach().cpu() for name, value in model.state_dict().items()
                 if value.requires_grad or name.startswith((
                     "paper_queries", "risk_branch.", "protective_branch.",
                     "risk_residual.", "protective_residual.",
                     "definition_gate", "risk_gate", "protective_gate",
                 ))}
        torch.save({"training_version": TRAINING_VERSION, "state_dict": state}, CHECKPOINT)
        np.savez_compressed(
            PREDICTIONS, probabilities=probability, valid_indices=valid_idx,
            history=json.dumps(history), training_version=TRAINING_VERSION,
        )
        del model; torch.cuda.empty_cache()

    base_saved = np.load(BASE_OOF)
    accepted_semantic = base_saved["semantic"][valid_idx].astype(np.float32)
    current, calibration = _current_v3_probability()
    component_weight = float(calibration["base_weight"]) * float(config.FACTOR_SEMANTIC_MODEL_WEIGHT)
    candidate_probability = (
        current[valid_idx] + component_weight * REPLACEMENT_WEIGHT
        * (probability - accepted_semantic)
    )
    truth = targets[valid_idx].astype(np.int8); prevalence = targets[train_idx].mean(0)
    baseline_prediction = _rank_decode(current[valid_idx], prevalence, TOPK_RATIO)
    candidate_prediction = _rank_decode(candidate_probability, prevalence, TOPK_RATIO)
    standalone_prediction = _rank_decode(probability, prevalence, TOPK_RATIO)
    baseline = float(f1_score(truth, baseline_prediction, average="macro", zero_division=0))
    candidate = float(f1_score(truth, candidate_prediction, average="macro", zero_division=0))
    standalone = float(f1_score(truth, standalone_prediction, average="macro", zero_division=0))
    bootstrap = _user_bootstrap(
        truth, baseline_prediction, candidate_prediction, groups[valid_idx],
        seed=363636, draws=3000,
    )
    payload = {
        "training_version": TRAINING_VERSION,
        "paper_alignment": {
            "source": "Li et al. (2025), Table III and Eq. (6)-(9)",
            "published_definitions": len(PAPER_FACTOR_DEFINITIONS),
            "independent_risk_protective_branches": True,
            "accepted_encoder_and_heads_frozen": True,
        },
        "evaluation": "untouched outer user fold0; fixed training and replacement",
        "history": history,
        "fixed_policy": {
            "epochs": EPOCHS, "replacement_weight": REPLACEMENT_WEIGHT,
            "effective_total_weight": component_weight * REPLACEMENT_WEIGHT,
            "topk_ratio": TOPK_RATIO,
        },
        "accepted_semantic_auc": _macro_auc(truth, accepted_semantic),
        "dual_branch_auc": _macro_auc(truth, probability),
        "dual_branch_standalone_macro_f1": standalone,
        "baseline_macro_f1": baseline, "candidate_macro_f1": candidate,
        "delta": candidate - baseline, "user_cluster_bootstrap": bootstrap,
        "promising_for_full_oof": bool(
            candidate >= baseline + .005 and bootstrap["positive_fraction"] >= .70
            and _macro_auc(truth, probability) >= _macro_auc(truth, accepted_semantic)
        ),
        "adopted": False,
    }
    RESULTS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    return payload


if __name__ == "__main__":
    train_fold0()
