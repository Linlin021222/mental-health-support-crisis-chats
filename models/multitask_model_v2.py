"""Experimental Task 1 model with ordinal risk and token-rationale heads."""
import math
import torch
import torch.nn as nn

from configs.config import config
from models.backbone import DebertaBackbone
from models.pooling import DocumentPooling
from models.heads import RiskClassificationHead, EvidenceExtractionHead, FactorClassificationHead


class OrderedRiskHead(nn.Module):
    """CORAL-style cumulative logits with guaranteed ordered cut-points."""
    def __init__(self, hidden_size=config.HIDDEN_SIZE):
        super().__init__()
        self.dropout = nn.Dropout(0.2)
        self.score = nn.Linear(hidden_size, 1, bias=False)
        self.first_bias = nn.Parameter(torch.tensor(1.0))
        # softplus(0.5413) is approximately one, giving [1, 0, -1].
        self.raw_gaps = nn.Parameter(torch.full((2,), 0.5413))

    def forward(self, representation):
        score = self.score(self.dropout(representation))
        gaps = torch.nn.functional.softplus(self.raw_gaps)
        biases = torch.stack([
            self.first_bias,
            self.first_bias - gaps[0],
            self.first_bias - gaps.sum(),
        ])
        return score + biases.unsqueeze(0)


class SuicideRiskMultiTaskModelV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = DebertaBackbone()
        self.pooling = DocumentPooling()
        self.risk_head = RiskClassificationHead()
        self.ordinal_risk_head = OrderedRiskHead()
        self.evidence_head = EvidenceExtractionHead()
        self.token_evidence_head = nn.Sequential(
            nn.Dropout(0.1), nn.Linear(config.HIDDEN_SIZE, 1)
        )
        self.factor_head = FactorClassificationHead()

    def forward(self, input_ids, attention_mask):
        token_hidden = self.backbone(input_ids, attention_mask).float()
        document_repr = self.pooling(token_hidden, attention_mask)
        start_logits, end_logits = self.evidence_head(token_hidden)
        return {
            "risk_logits": self.risk_head(document_repr),
            "ordinal_logits": self.ordinal_risk_head(document_repr),
            "start_logits": start_logits,
            "end_logits": end_logits,
            "token_logits": self.token_evidence_head(token_hidden).squeeze(-1),
            "factor_logits": self.factor_head(token_hidden, attention_mask, document_repr),
        }


def ordinal_class_probabilities(ordinal_logits):
    """Convert monotone P(y>k) values into four class probabilities."""
    cumulative = torch.sigmoid(ordinal_logits)
    return torch.stack([
        1.0 - cumulative[:, 0],
        cumulative[:, 0] - cumulative[:, 1],
        cumulative[:, 1] - cumulative[:, 2],
        cumulative[:, 2],
    ], dim=-1).clamp_min(0.0)


def v2_optimizer_parameters(model):
    backbone, heads = [], []
    for name, parameter in model.named_parameters():
        (backbone if name.startswith("backbone.") else heads).append(parameter)
    return [
        {"params": backbone, "lr": config.BACKBONE_LR},
        {"params": heads, "lr": config.HEAD_LR},
    ]
