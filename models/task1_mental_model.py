"""Domain-adapted MentalRoBERTa expert for Task 1 risk classification."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from configs.config import config


RISK_DESCRIPTIONS = (
    "The post has no explicit mention of suicide, wanting to die, a suicide method, or a suicide attempt.",
    "The author explicitly wants to die or expresses suicidal thoughts, but gives no suicide plan.",
    "The author expresses suicide and describes a plan, method, or access to suicide means.",
    "The author explicitly describes a past or recent suicide attempt.",
)


def ordinal_class_probabilities(logits):
    """Convert P(y > k), k=0..2, into a four-class distribution."""
    cumulative = torch.sigmoid(logits)
    cumulative = torch.cummin(cumulative, dim=-1).values
    probability = torch.stack((
        1.0 - cumulative[:, 0],
        cumulative[:, 0] - cumulative[:, 1],
        cumulative[:, 1] - cumulative[:, 2],
        cumulative[:, 2],
    ), dim=-1)
    return probability.clamp_min(1e-7) / probability.sum(-1, keepdim=True).clamp_min(1e-7)


class MentalRobertaRiskModel(nn.Module):
    """Label-attentive categorical/ordinal classifier over long Reddit posts."""
    def __init__(self, initialise_labels=True):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(config.FACTOR_MODEL_NAME, dtype=torch.float32)
        # The 8 GB target GPU has enough memory for two 384-token windows.
        # Checkpointing made this small-data experiment roughly four times
        # slower, so keep activations and use normal backpropagation.
        self.encoder.config.use_cache = False
        hidden = self.encoder.config.hidden_size
        self.norm = nn.LayerNorm(hidden)
        if initialise_labels:
            label_vectors = self._description_vectors()
        else:
            label_vectors = torch.empty(config.NUM_RISK_CLASSES, hidden)
            nn.init.xavier_uniform_(label_vectors)
        self.label_queries = nn.Parameter(label_vectors.clone())
        self.label_weights = nn.Parameter(
            torch.nn.functional.normalize(label_vectors.clone(), dim=-1)
        )
        self.label_bias = nn.Parameter(torch.zeros(config.NUM_RISK_CLASSES))
        self.global_classifier = nn.Linear(hidden, config.NUM_RISK_CLASSES)
        self.ordinal_head = nn.Linear(hidden, config.NUM_RISK_CLASSES - 1)
        self.dropout = nn.Dropout(0.15)

    def _description_vectors(self):
        tokenizer = AutoTokenizer.from_pretrained(config.FACTOR_MODEL_NAME, use_fast=True)
        encoded = tokenizer(
            RISK_DESCRIPTIONS, padding=True, truncation=True, max_length=64,
            return_tensors="pt",
        )
        was_training = self.encoder.training
        self.encoder.eval()
        with torch.no_grad():
            hidden = self.encoder(**encoded).last_hidden_state.detach().cpu()
        if was_training:
            self.encoder.train()
        mask = encoded["attention_mask"].unsqueeze(-1).float()
        return (hidden * mask).sum(1) / mask.sum(1).clamp_min(1.0)

    def forward(self, input_ids, attention_mask):
        batch, chunks, length = input_ids.shape
        flat_ids = input_ids.reshape(batch * chunks, length)
        flat_mask = attention_mask.reshape(batch * chunks, length)
        hidden = self.encoder(input_ids=flat_ids, attention_mask=flat_mask).last_hidden_state
        hidden = self.norm(hidden.float().reshape(batch, chunks * length, -1))
        mask = attention_mask.reshape(batch, chunks * length).bool()

        scores = torch.einsum("bth,kh->btk", hidden, self.label_queries)
        scores = scores / math.sqrt(hidden.size(-1))
        scores = scores.masked_fill(~mask.unsqueeze(-1), -1e4)
        attention = torch.softmax(scores, dim=1)
        label_repr = torch.einsum("btk,bth->bkh", attention, hidden)
        local = (
            self.dropout(label_repr) * self.label_weights.unsqueeze(0)
        ).sum(-1) + self.label_bias

        mask_float = mask.unsqueeze(-1).to(hidden.dtype)
        global_repr = (hidden * mask_float).sum(1) / mask_float.sum(1).clamp_min(1.0)
        risk_logits = local + self.global_classifier(self.dropout(global_repr))
        ordinal_logits = self.ordinal_head(self.dropout(global_repr))
        return {"risk_logits": risk_logits, "ordinal_logits": ordinal_logits}


def optimizer_parameters(model, backbone_lr=6e-6, head_lr=3e-5):
    backbone, head = [], []
    for name, parameter in model.named_parameters():
        (backbone if name.startswith("encoder.") else head).append(parameter)
    return [
        {"params": backbone, "lr": float(backbone_lr)},
        {"params": head, "lr": float(head_lr)},
    ]
