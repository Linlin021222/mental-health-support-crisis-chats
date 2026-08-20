"""Standalone domain-adapted MentalRoBERTa model for Subtask 2."""
import math
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from configs.config import config


class MentalRobertaFactorModel(nn.Module):
    def __init__(self, initialise_labels=True):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(config.FACTOR_MODEL_NAME, dtype=torch.float32)
        self.encoder.gradient_checkpointing_enable()
        self.encoder.config.use_cache = False
        hidden = self.encoder.config.hidden_size
        self.norm = nn.LayerNorm(hidden)
        if initialise_labels:
            label_queries = self._semantic_label_initialisation(hidden)
        else:
            # Inference immediately loads this tensor from a checkpoint, so
            # avoid five redundant CPU encoder passes for the CV ensemble.
            label_queries = torch.empty(config.NUM_FACTORS, hidden)
            nn.init.xavier_uniform_(label_queries)
        self.label_queries = nn.Parameter(label_queries)
        if initialise_labels and config.FACTOR_SEMANTIC_CLASSIFIER_INIT:
            # The former implementation only initialised attention queries;
            # the actual classifiers remained random, which largely erased
            # the benefit for labels with 8--20 positives.  Initialise both
            # sides of the label-specific detector from the taxonomy.
            label_weights = torch.nn.functional.normalize(label_queries.clone(), dim=-1)
        else:
            label_weights = torch.empty(config.NUM_FACTORS, hidden)
            nn.init.xavier_uniform_(label_weights)
        self.label_weights = nn.Parameter(label_weights)
        self.label_bias = nn.Parameter(torch.zeros(config.NUM_FACTORS))
        self.global_risk = nn.Linear(hidden, 19)
        self.global_protective = nn.Linear(hidden, 5)
        self.dropout = nn.Dropout(0.15)

    def _semantic_label_initialisation(self, hidden):
        tokenizer = AutoTokenizer.from_pretrained(config.FACTOR_MODEL_NAME, use_fast=True)
        descriptions = (config.FACTOR_NLI_HYPOTHESES
                        if config.FACTOR_PAPER_DEFINITION_INIT
                        else config.FACTOR_DESCRIPTIONS)
        tokenized = tokenizer(
            descriptions, padding=True, truncation=True,
            max_length=64, return_tensors="pt",
        )
        if config.FACTOR_CONTEXTUAL_LABEL_INIT:
            # CTN-LT/dual-encoder style initialisation: encode the complete
            # natural-language taxonomy description rather than averaging
            # isolated input embeddings. This gives rare labels a contextual
            # semantic prototype before seeing their few supervised examples.
            was_training = self.encoder.training
            self.encoder.eval()
            with torch.no_grad():
                encoded = self.encoder(**tokenized).last_hidden_state.detach().cpu()
            if was_training:
                self.encoder.train()
            mask = tokenized["attention_mask"].unsqueeze(-1).float()
            values = (encoded * mask).sum(1) / mask.sum(1).clamp_min(1.0)
        else:
            embedding = self.encoder.get_input_embeddings().weight.detach().cpu()
            vectors = []
            special = set(tokenizer.all_special_ids)
            for ids in tokenized["input_ids"]:
                keep = torch.tensor([int(x) not in special for x in ids], dtype=torch.bool)
                vectors.append(embedding[ids[keep]].mean(0))
            values = torch.stack(vectors)
        if values.shape != (config.NUM_FACTORS, hidden):
            raise ValueError(f"Bad semantic label initialisation shape: {values.shape}")
        return values

    def forward(self, input_ids, attention_mask, return_semantic=False,
                return_features=False):
        batch, chunks, length = input_ids.shape
        flat_ids = input_ids.reshape(batch * chunks, length)
        flat_mask = attention_mask.reshape(batch * chunks, length)
        hidden = self.encoder(input_ids=flat_ids, attention_mask=flat_mask).last_hidden_state
        hidden = hidden.float().reshape(batch, chunks * length, -1)
        mask = attention_mask.reshape(batch, chunks * length).bool()
        tokens = self.norm(hidden)
        scores = torch.einsum("bth,kh->btk", tokens, self.label_queries)
        scores = scores / math.sqrt(tokens.size(-1))
        scores = scores.masked_fill(~mask.unsqueeze(-1), -1e4)
        attention = torch.softmax(scores, dim=1)
        label_repr = torch.einsum("btk,bth->bkh", attention, tokens)
        local = (self.dropout(label_repr) * self.label_weights.unsqueeze(0)).sum(-1) + self.label_bias
        mask_float = mask.unsqueeze(-1).to(tokens.dtype)
        global_repr = (tokens * mask_float).sum(1) / mask_float.sum(1).clamp_min(1.0)
        global_logits = torch.cat([self.global_risk(global_repr), self.global_protective(global_repr)], dim=-1)
        logits = local + global_logits
        if not return_semantic and not return_features:
            return logits
        # CTN-LT-style text/label semantic alignment. The label queries were
        # initialised from natural-language taxonomy definitions; this
        # auxiliary objective keeps the document representation aligned with
        # those definitions, including for rare labels.
        document_semantic = torch.nn.functional.normalize(global_repr, dim=-1)
        label_semantic = torch.nn.functional.normalize(self.label_queries, dim=-1)
        semantic_logits = torch.einsum("bh,kh->bk", document_semantic, label_semantic)
        semantic_logits = semantic_logits / config.FACTOR_SEMANTIC_TEMPERATURE
        if return_semantic and return_features:
            return logits, semantic_logits, global_repr
        if return_semantic:
            return logits, semantic_logits
        return logits, global_repr


def factor_optimizer_parameters(model):
    backbone, head = [], []
    for name, parameter in model.named_parameters():
        (backbone if name.startswith("encoder.") else head).append(parameter)
    return [
        {"params": backbone, "lr": config.BACKBONE_LR},
        {"params": head, "lr": config.HEAD_LR},
    ]
