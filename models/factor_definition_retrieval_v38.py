"""Definition-grounded, long-tail and retrieval-augmented MentalRoBERTa."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

from configs.config import config
from models.factor_model import MentalRobertaFactorModel
from preprocess.factor_paper_definitions_v36 import PAPER_FACTOR_DEFINITIONS


class DefinitionRetrievalFactorModel(MentalRobertaFactorModel):
    def __init__(self, definition_embeddings, tail_mask, feature_std):
        super().__init__(initialise_labels=False)
        hidden = self.encoder.config.hidden_size
        self.register_buffer("definition_embeddings", definition_embeddings.float())
        self.register_buffer("tail_mask", tail_mask.float())
        self.register_buffer("feature_std", feature_std.float())
        self.definition_projection = nn.Linear(hidden, hidden, bias=False)
        self.definition_gate = nn.Parameter(torch.full((config.NUM_FACTORS,), -2.2))
        self.retrieval_projection = nn.Linear(hidden, hidden, bias=False)
        self.retrieval_gate = nn.Parameter(torch.full((config.NUM_FACTORS,), -2.2))
        self.fusion_norm = nn.LayerNorm(hidden)
        # HTTN-style classifier generator: learn the classifier mapping on
        # head labels and apply the residual only to tail labels.
        self.tail_generator = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, hidden),
        )
        nn.init.zeros_(self.tail_generator[-1].weight)
        nn.init.zeros_(self.tail_generator[-1].bias)
        self.tail_gate = nn.Parameter(torch.tensor(-2.2))

    def forward(self, input_ids, attention_mask, retrieval_features,
                return_aux=False):
        batch, chunks, length = input_ids.shape
        flat_ids = input_ids.reshape(batch * chunks, length)
        flat_mask = attention_mask.reshape(batch * chunks, length)
        hidden = self.encoder(input_ids=flat_ids, attention_mask=flat_mask).last_hidden_state
        hidden = hidden.float().reshape(batch, chunks * length, -1)
        mask = attention_mask.reshape(batch, chunks * length).bool()
        tokens = self.norm(hidden)

        definitions = self.definition_projection(self.definition_embeddings)
        query = self.label_queries + torch.sigmoid(self.definition_gate).unsqueeze(-1) * definitions
        scores = torch.einsum("bth,kh->btk", tokens, query) / math.sqrt(tokens.size(-1))
        scores = scores.masked_fill(~mask.unsqueeze(-1), -1e4)
        attention = torch.softmax(scores, dim=1)
        label_repr = torch.einsum("btk,bth->bkh", attention, tokens)

        retrieval = self.retrieval_projection(retrieval_features.float())
        retrieval_score = torch.einsum(
            "bkh,bnh->bkn", F.normalize(label_repr, dim=-1), F.normalize(retrieval, dim=-1)
        )
        retrieval_attention = torch.softmax(retrieval_score, dim=-1)
        retrieved = torch.einsum("bkn,bnh->bkh", retrieval_attention, retrieval)
        fused = self.fusion_norm(
            label_repr + torch.sigmoid(self.retrieval_gate).view(1, -1, 1) * retrieved
        )

        generated = self.tail_generator(self.definition_embeddings)
        effective_weight = self.label_weights + (
            torch.sigmoid(self.tail_gate) * self.tail_mask.unsqueeze(-1) * generated
        )
        local = (self.dropout(fused) * effective_weight.unsqueeze(0)).sum(-1) + self.label_bias
        mask_float = mask.unsqueeze(-1).to(tokens.dtype)
        global_repr = (tokens * mask_float).sum(1) / mask_float.sum(1).clamp_min(1.)
        global_logits = torch.cat(
            (self.global_risk(global_repr), self.global_protective(global_repr)), dim=-1,
        )
        logits = local + global_logits
        if not return_aux:
            return logits

        definition_logits = torch.einsum(
            "bkh,jh->bkj", F.normalize(fused, dim=-1),
            F.normalize(self.definition_embeddings, dim=-1),
        ) / .12
        semantic_logits = torch.einsum(
            "bh,kh->bk", F.normalize(global_repr, dim=-1),
            F.normalize(self.definition_embeddings, dim=-1),
        ) / .12
        # LSFA-style positive pair augmentation transfers high-level feature
        # variation to tail label representations without fabricating text or
        # changing any co-occurring labels.
        augmented = fused + torch.randn_like(fused) * self.feature_std.view(1, 1, -1) * .15
        augmented_logits = (
            augmented * effective_weight.unsqueeze(0)
        ).sum(-1) + self.label_bias + global_logits
        return logits, {
            "definition_logits": definition_logits,
            "semantic_logits": semantic_logits,
            "generated_weights": generated,
            "effective_weights": effective_weight,
            "augmented_logits": augmented_logits,
        }


@torch.no_grad()
def encode_paper_definitions(model, device):
    tokenizer = AutoTokenizer.from_pretrained(config.FACTOR_MODEL_NAME, use_fast=True)
    encoded = tokenizer(
        PAPER_FACTOR_DEFINITIONS, padding=True, truncation=True,
        max_length=96, return_tensors="pt",
    ).to(device)
    was_training = model.encoder.training; model.encoder.eval()
    hidden = model.encoder(**encoded).last_hidden_state.float()
    mask = encoded["attention_mask"].unsqueeze(-1).float()
    values = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1.)
    if was_training:
        model.encoder.train()
    return values.detach()


__all__ = ["DefinitionRetrievalFactorModel", "encode_paper_definitions"]
