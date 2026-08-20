"""Independent DeBERTa-v3 factor ranker for heterogeneous ensembling."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from configs.config import config


class DebertaFactorExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(
            config.MODEL_NAME, dtype=torch.float32, local_files_only=True,
        )
        self.encoder.gradient_checkpointing_enable(); self.encoder.config.use_cache = False
        hidden = int(self.encoder.config.hidden_size)
        tokenizer = AutoTokenizer.from_pretrained(
            config.MODEL_NAME, use_fast=True, local_files_only=True,
        )
        definitions = tokenizer(
            config.FACTOR_NLI_HYPOTHESES, padding=True, truncation=True,
            max_length=96, return_tensors="pt",
        )
        was_training = self.encoder.training; self.encoder.eval()
        with torch.no_grad():
            values = self.encoder(**definitions).last_hidden_state.float()
        if was_training: self.encoder.train()
        definition_mask = definitions["attention_mask"].unsqueeze(-1).float()
        queries = (values * definition_mask).sum(1) / definition_mask.sum(1).clamp_min(1.)
        self.norm = nn.LayerNorm(hidden)
        self.label_queries = nn.Parameter(queries.detach())
        self.label_weights = nn.Parameter(torch.empty(config.NUM_FACTORS, hidden))
        nn.init.xavier_uniform_(self.label_weights)
        self.label_bias = nn.Parameter(torch.zeros(config.NUM_FACTORS))
        self.risk_global = nn.Sequential(nn.LayerNorm(hidden), nn.Dropout(.15), nn.Linear(hidden, 19))
        self.protective_global = nn.Sequential(nn.LayerNorm(hidden), nn.Dropout(.15), nn.Linear(hidden, 5))
        self.dropout = nn.Dropout(.15)

    def forward(self, input_ids, attention_mask, return_semantic=False):
        batch, chunks, length = input_ids.shape
        ids = input_ids.reshape(batch * chunks, length)
        flat_mask = attention_mask.reshape(batch * chunks, length)
        hidden = self.encoder(input_ids=ids, attention_mask=flat_mask).last_hidden_state
        tokens = self.norm(hidden.float().reshape(batch, chunks * length, -1))
        mask = attention_mask.reshape(batch, chunks * length).bool()
        score = torch.einsum("bth,kh->btk", tokens, self.label_queries)
        score = score / math.sqrt(tokens.size(-1))
        score = score.masked_fill(~mask.unsqueeze(-1), -1e4)
        attention = torch.softmax(score, dim=1)
        label_repr = torch.einsum("btk,bth->bkh", attention, tokens)
        local = (self.dropout(label_repr) * self.label_weights.unsqueeze(0)).sum(-1) + self.label_bias
        float_mask = mask.unsqueeze(-1).to(tokens.dtype)
        document = (tokens * float_mask).sum(1) / float_mask.sum(1).clamp_min(1.)
        logits = local + torch.cat((self.risk_global(document), self.protective_global(document)), -1)
        if not return_semantic: return logits
        semantic = torch.einsum(
            "bh,kh->bk", F.normalize(document, dim=-1),
            F.normalize(self.label_queries, dim=-1),
        ) / .12
        return logits, semantic


__all__ = ["DebertaFactorExpert"]
