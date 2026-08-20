"""Losses for risk classification, evidence extraction and factor labels."""
import torch
import torch.nn as nn
from configs.config import config


class RiskLoss(nn.Module):
    def __init__(self, class_weights=None):
        super().__init__()
        self.loss = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, logits, labels):
        return self.loss(logits, labels)


class EvidenceLoss(nn.Module):
    def __init__(self, pos_weight=config.EVIDENCE_POS_WEIGHT):
        super().__init__()
        self.pos_weight = float(pos_weight)

    def _side(self, logits, labels, attention_mask):
        loss = nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="none")
        weights = torch.where(labels > 0, self.pos_weight, 1.0).to(loss.dtype)
        mask = attention_mask.to(loss.dtype)
        return (loss * weights * mask).sum() / mask.sum().clamp_min(1.0)

    def forward(self, start_logits, end_logits, start_labels, end_labels, attention_mask):
        return 0.5 * (self._side(start_logits, start_labels, attention_mask) +
                      self._side(end_logits, end_labels, attention_mask))


class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=config.ASL_GAMMA_NEG, gamma_pos=config.ASL_GAMMA_POS,
                 clip=config.ASL_CLIP, eps=1e-8):
        super().__init__()
        self.gamma_neg, self.gamma_pos, self.clip, self.eps = gamma_neg, gamma_pos, clip, eps

    def forward(self, logits, targets):
        xs_pos = torch.sigmoid(logits)
        xs_neg = 1 - xs_pos
        if self.clip is not None:
            xs_neg = (xs_neg + self.clip).clamp(max=1)
        loss = targets * torch.log(xs_pos.clamp(min=self.eps))
        loss += (1 - targets) * torch.log(xs_neg.clamp(min=self.eps))
        pt = xs_pos * targets + xs_neg * (1 - targets)
        gamma = self.gamma_pos * targets + self.gamma_neg * (1 - targets)
        return -(loss * torch.pow(1 - pt, gamma)).mean()


class FactorLoss(nn.Module):
    def __init__(self, pos_weight=None):
        super().__init__()
        if pos_weight is None:
            self.register_buffer("pos_weight", torch.ones(config.NUM_FACTORS))
        else:
            self.register_buffer("pos_weight", pos_weight.detach().clone())
        self.asymmetric = AsymmetricLoss()

    def _group_loss(self, logits, labels, pos_weight):
        if config.FACTOR_LOSS == "asymmetric":
            return self.asymmetric(logits, labels)
        return nn.functional.binary_cross_entropy_with_logits(
            logits, labels, pos_weight=pos_weight
        )

    def forward(self, logits, labels):
        # The source taxonomy contains 19 risk factors and five protective
        # factors. Give the two groups equal loss mass so the smaller
        # protective group is not overwhelmed by the 19 risk outputs.
        risk = self._group_loss(logits[:, :19], labels[:, :19], self.pos_weight[:19])
        protective = self._group_loss(logits[:, 19:], labels[:, 19:], self.pos_weight[19:])
        return 0.5 * (risk + protective)


class MultiTaskLoss(nn.Module):
    def __init__(self, weights=None, risk_class_weights=None, factor_pos_weight=None):
        super().__init__()
        self.risk_loss = RiskLoss(risk_class_weights)
        self.evidence_loss = EvidenceLoss()
        self.factor_loss = FactorLoss(factor_pos_weight)
        self.weights = config.LOSS_WEIGHTS if weights is None else weights

    def forward(self, outputs, labels):
        risk = self.risk_loss(outputs["risk_logits"], labels["risk_labels"])
        evidence = self.evidence_loss(outputs["start_logits"], outputs["end_logits"],
                                      labels["start_labels"], labels["end_labels"], labels["attention_mask"])
        factor = self.factor_loss(outputs["factor_logits"], labels["factor_vectors"])
        total = self.weights["risk"] * risk + self.weights["evidence"] * evidence + self.weights["factor"] * factor
        return {"loss": total, "risk_loss": risk, "evidence_loss": evidence, "factor_loss": factor}
