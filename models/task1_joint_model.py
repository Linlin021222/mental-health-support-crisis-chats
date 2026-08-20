"""Task 1-only DeBERTa with nominal, ordinal, boundary, and token evidence heads."""
import torch.nn as nn

from configs.config import config
from models.backbone import DebertaBackbone
from models.heads import EvidenceExtractionHead, RiskClassificationHead
from models.multitask_model_v2 import OrderedRiskHead, ordinal_class_probabilities
from models.pooling import DocumentPooling


class Task1JointModel(nn.Module):
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

    def forward(self, input_ids, attention_mask):
        token_hidden = self.backbone(input_ids, attention_mask).float()
        document = self.pooling(token_hidden, attention_mask)
        start, end = self.evidence_head(token_hidden)
        output = {
            "risk_logits": self.risk_head(document),
            "ordinal_logits": self.ordinal_risk_head(document),
            "start_logits": start,
            "end_logits": end,
            "token_logits": self.token_evidence_head(token_hidden).squeeze(-1),
        }
        # Low-cost R-Drop: reuse the expensive encoder representation and run
        # only the stochastic risk heads twice.  The symmetric consistency
        # term is applied by JointTask1Loss during training.
        if self.training and config.TASK1_HEAD_RDROP_WEIGHT > 0:
            output["risk_logits_rdrop"] = self.risk_head(document)
            output["ordinal_logits_rdrop"] = self.ordinal_risk_head(document)
        return output


def optimizer_parameters(model):
    groups = {}
    layer_count = len(model.backbone.encoder.encoder.layer)
    for name, parameter in model.named_parameters():
        if not name.startswith("backbone."):
            learning_rate = config.HEAD_LR
        elif ".encoder.layer." in name:
            layer = int(name.split(".encoder.layer.", 1)[1].split(".", 1)[0])
            depth = layer_count - 1 - layer
            learning_rate = config.BACKBONE_LR * config.TASK1_LAYERWISE_LR_DECAY ** depth
        else:
            # Embeddings and any shared encoder parameters receive the most
            # conservative rate; upper layers remain free to adapt.
            learning_rate = config.BACKBONE_LR * config.TASK1_LAYERWISE_LR_DECAY ** layer_count
        groups.setdefault(float(learning_rate), []).append(parameter)
    return [{"params": parameters, "lr": lr} for lr, parameters in groups.items()]


__all__ = ["Task1JointModel", "ordinal_class_probabilities", "optimizer_parameters"]
