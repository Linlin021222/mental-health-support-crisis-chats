"""Paper-aligned current-post multi-task factor model.

Only the components of Li et al. (2025) that match the competition are
transferred: independent risk/protective representations and joint learning
with an ordinal suicide-risk auxiliary task.  The paper's next-post dynamic
transition objective is deliberately omitted because the leaderboard rows are
independent current posts and do not provide a subsequent-risk target.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from configs.config import config
from models.factor_model import MentalRobertaFactorModel


class PFAJointFactorModel(MentalRobertaFactorModel):
    """Accepted MentalRoBERTa plus separate RF/PF and risk auxiliary heads."""

    def __init__(self):
        super().__init__(initialise_labels=False)
        hidden = int(self.encoder.config.hidden_size)
        self.risk_query_delta = nn.Parameter(torch.zeros(hidden))
        self.protective_query_delta = nn.Parameter(torch.zeros(hidden))
        self.risk_branch = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, hidden // 2), nn.GELU(),
            nn.Dropout(.12), nn.Linear(hidden // 2, hidden), nn.GELU(),
        )
        self.protective_branch = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, hidden // 2), nn.GELU(),
            nn.Dropout(.12), nn.Linear(hidden // 2, hidden), nn.GELU(),
        )
        self.risk_factor_residual = nn.Linear(hidden, 19)
        self.protective_factor_residual = nn.Linear(hidden, 5)
        # Zero residuals make the loaded fold checkpoint exactly reproducible
        # before continuation training.
        nn.init.zeros_(self.risk_factor_residual.weight)
        nn.init.zeros_(self.risk_factor_residual.bias)
        nn.init.zeros_(self.protective_factor_residual.weight)
        nn.init.zeros_(self.protective_factor_residual.bias)
        self.risk_factor_gate = nn.Parameter(torch.tensor(-2.2))
        self.protective_factor_gate = nn.Parameter(torch.tensor(-2.2))
        self.risk_level_head = nn.Sequential(
            nn.LayerNorm(hidden * 3), nn.Linear(hidden * 3, hidden // 2),
            nn.GELU(), nn.Dropout(.15), nn.Linear(hidden // 2, 4),
        )
        # Kendall et al. uncertainty weights used by Eq. (18) of the paper.
        self.task_log_vars = nn.Parameter(torch.zeros(3))  # RF, PF, SR

    @staticmethod
    def _branch_pool(tokens, mask, query):
        score = torch.einsum("bth,h->bt", tokens, query) / math.sqrt(tokens.size(-1))
        score = score.masked_fill(~mask, -1e4)
        attention = torch.softmax(score, dim=1)
        return torch.einsum("bt,bth->bh", attention, tokens)

    def forward(self, input_ids, attention_mask, return_aux=False):
        batch, chunks, length = input_ids.shape
        flat_ids = input_ids.reshape(batch * chunks, length)
        flat_mask = attention_mask.reshape(batch * chunks, length)
        hidden = self.encoder(input_ids=flat_ids, attention_mask=flat_mask).last_hidden_state
        tokens = self.norm(hidden.float().reshape(batch, chunks * length, -1))
        mask = attention_mask.reshape(batch, chunks * length).bool()

        # Existing label-specific attention and classifiers are retained.
        score = torch.einsum("bth,kh->btk", tokens, self.label_queries)
        score = score / math.sqrt(tokens.size(-1))
        score = score.masked_fill(~mask.unsqueeze(-1), -1e4)
        label_attention = torch.softmax(score, dim=1)
        label_repr = torch.einsum("btk,bth->bkh", label_attention, tokens)
        local = (
            self.dropout(label_repr) * self.label_weights.unsqueeze(0)
        ).sum(-1) + self.label_bias
        float_mask = mask.unsqueeze(-1).to(tokens.dtype)
        global_repr = (tokens * float_mask).sum(1) / float_mask.sum(1).clamp_min(1.)
        accepted_global = torch.cat((
            self.global_risk(global_repr), self.global_protective(global_repr),
        ), dim=-1)

        # The two query anchors are derived from the semantic label bank, but
        # have independent learnable corrections and independent MLP spaces.
        risk_query = self.label_queries[:19].mean(0) + self.risk_query_delta
        protective_query = self.label_queries[19:].mean(0) + self.protective_query_delta
        risk_repr = self.risk_branch(self._branch_pool(tokens, mask, risk_query))
        protective_repr = self.protective_branch(
            self._branch_pool(tokens, mask, protective_query)
        )
        residual = torch.cat((
            torch.sigmoid(self.risk_factor_gate) * self.risk_factor_residual(risk_repr),
            torch.sigmoid(self.protective_factor_gate)
            * self.protective_factor_residual(protective_repr),
        ), dim=-1)
        factor_logits = local + accepted_global + residual
        risk_level_logits = self.risk_level_head(torch.cat((
            global_repr, risk_repr, protective_repr,
        ), dim=-1))
        if not return_aux:
            return factor_logits
        return factor_logits, {
            "risk_level_logits": risk_level_logits,
            "risk_repr": risk_repr,
            "protective_repr": protective_repr,
            "task_log_vars": self.task_log_vars,
        }


__all__ = ["PFAJointFactorModel"]
