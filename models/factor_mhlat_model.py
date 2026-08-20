"""Two-hop label-aware MentalRoBERTa for the optional Task 2 v4 expert.

The module deliberately keeps all production-v3 parameter names compatible
with :class:`MentalRobertaFactorModel`.  Loading a completed factor-CV fold
therefore transfers the encoder, first-hop attention and classifiers; only the
small refinement hop starts new.  Its residual gate is initialised close to
zero so epoch zero behaves like the validated one-hop model.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.config import config
from models.factor_model import MentalRobertaFactorModel


class MentalRobertaMHLATModel(MentalRobertaFactorModel):
    """MHLAT-style second reading hop with learnable semantic label centres."""

    def __init__(self, initialise_labels=False):
        super().__init__(initialise_labels=initialise_labels)
        hidden = self.encoder.config.hidden_size
        self.hop_projection = nn.Linear(hidden, hidden)
        # A very small initial residual makes the transferred checkpoint the
        # actual baseline, instead of comparing it with a random new network.
        nn.init.normal_(self.hop_projection.weight, mean=0.0, std=0.002)
        nn.init.zeros_(self.hop_projection.bias)
        self.hop_gate = nn.Parameter(torch.full((config.NUM_FACTORS,), -1.75))
        self.hop_norm = nn.LayerNorm(hidden)

    def forward(self, input_ids, attention_mask, return_aux=False):
        batch, chunks, length = input_ids.shape
        flat_ids = input_ids.reshape(batch * chunks, length)
        flat_mask = attention_mask.reshape(batch * chunks, length)
        hidden = self.encoder(input_ids=flat_ids, attention_mask=flat_mask).last_hidden_state
        tokens = self.norm(hidden.float().reshape(batch, chunks * length, -1))
        mask = attention_mask.reshape(batch, chunks * length).bool()
        scale = math.sqrt(tokens.size(-1))

        first_scores = torch.einsum("bth,kh->btk", tokens, self.label_queries) / scale
        first_scores = first_scores.masked_fill(~mask.unsqueeze(-1), -1e4)
        first_attention = torch.softmax(first_scores, dim=1)
        first_repr = torch.einsum("btk,bth->bkh", first_attention, tokens)

        # Each label uses its first reading to form a sample-specific query and
        # then reads the token sequence again. This is the key MHLAT operation.
        refined_query = self.label_queries.unsqueeze(0) + torch.tanh(
            self.hop_projection(self.hop_norm(first_repr))
        )
        second_scores = torch.einsum("bth,bkh->btk", tokens, refined_query) / scale
        second_scores = second_scores.masked_fill(~mask.unsqueeze(-1), -1e4)
        second_attention = torch.softmax(second_scores, dim=1)
        second_repr = torch.einsum("btk,bth->bkh", second_attention, tokens)
        gate = torch.sigmoid(self.hop_gate).view(1, -1, 1)
        label_repr = first_repr + gate * (second_repr - first_repr)

        local = (
            self.dropout(label_repr) * self.label_weights.unsqueeze(0)
        ).sum(-1) + self.label_bias
        mask_float = mask.unsqueeze(-1).to(tokens.dtype)
        global_repr = (tokens * mask_float).sum(1) / mask_float.sum(1).clamp_min(1.0)
        global_logits = torch.cat(
            [self.global_risk(global_repr), self.global_protective(global_repr)], dim=-1
        )
        logits = local + global_logits
        if not return_aux:
            return logits

        document_semantic = F.normalize(global_repr, dim=-1)
        label_centres = F.normalize(self.label_queries, dim=-1)
        semantic_logits = torch.einsum("bh,kh->bk", document_semantic, label_centres)
        semantic_logits = semantic_logits / config.FACTOR_SEMANTIC_TEMPERATURE
        # [batch, source label, candidate centre]. Positive source-label
        # representations are contrasted with all 24 semantic label centres.
        centre_logits = torch.einsum(
            "bkh,jh->bkj", F.normalize(label_repr, dim=-1), label_centres
        ) / config.FACTOR_MHLAT_TEMPERATURE
        return logits, semantic_logits, centre_logits


def mhlat_optimizer_parameters(model):
    backbone, heads = [], []
    for name, parameter in model.named_parameters():
        (backbone if name.startswith("encoder.") else heads).append(parameter)
    return [
        {"params": backbone, "lr": config.FACTOR_MHLAT_BACKBONE_LR},
        {"params": heads, "lr": config.FACTOR_MHLAT_HEAD_LR},
    ]


__all__ = ["MentalRobertaMHLATModel", "mhlat_optimizer_parameters"]
