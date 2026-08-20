# ============================================================
# Suicide Risk Detection Competition
# Multi-task Prediction Heads
# ============================================================


import torch

import torch.nn as nn
import math


from configs.config import config





# ============================================================
# Risk Classification Head
# ============================================================


class RiskClassificationHead(nn.Module):

    """
    Subtask 1:

    Suicide risk level classification


    Input:

    document representation

    [B,H]


    Output:

    risk logits

    [B,4]

    """



    def __init__(
            self,
            hidden_size=config.HIDDEN_SIZE,
            num_classes=config.NUM_RISK_CLASSES
    ):


        super().__init__()



        self.dropout=nn.Dropout(
            0.2
        )



        self.classifier=nn.Linear(

            hidden_size,

            num_classes

        )





    def forward(
            self,
            x
    ):


        x=self.dropout(
            x
        )


        logits=self.classifier(
            x
        )


        return logits





# ============================================================
# Evidence Extraction Head
# ============================================================


class EvidenceExtractionHead(nn.Module):

    """
    Subtask 1:

    Evidence phrase extraction


    Input:

    token hidden states


    [B,C,L,H]


    Output:

    start logits:

    [B,C,L]


    end logits:

    [B,C,L]


    """



    def __init__(
            self,
            hidden_size=config.HIDDEN_SIZE
    ):


        super().__init__()



        self.dropout=nn.Dropout(
            0.1
        )



        self.span_classifier=nn.Linear(

            hidden_size,

            2

        )





    def forward(
            self,
            token_hidden
    ):



        """

        token_hidden:

        [B,C,L,H]


        """



        x=self.dropout(
            token_hidden
        )



        logits=self.span_classifier(
            x
        )



        # [B,C,L,2]



        start_logits=logits[...,0]



        end_logits=logits[...,1]



        return (

            start_logits,

            end_logits

        )





# ============================================================
# Factor Classification Head
# ============================================================


class FactorClassificationHead(nn.Module):

    """
    Subtask 2:

    Suicide factor multi-label classification


    Input:

    token representations, attention mask, and document representation


    Output:

    24 factor logits

    """



    def __init__(
            self,
            hidden_size=config.HIDDEN_SIZE,
            num_labels=config.NUM_FACTORS
    ):


        super().__init__()



        # Each taxonomy label learns its own token-attention query.  This is
        # better suited to a post that can simultaneously contain, for
        # example, hopelessness, family conflict, coping and social support.
        self.norm = nn.LayerNorm(hidden_size)
        self.num_risk_factors = num_labels - 5
        # Explicit label-specific queries. Each row is a learnable embedding
        # for one taxonomy label and cross-attends to every post token.
        self.label_queries = nn.Parameter(torch.empty(num_labels, hidden_size))
        nn.init.xavier_uniform_(self.label_queries)
        self.label_weight = nn.Parameter(torch.empty(num_labels, hidden_size))
        self.label_bias = nn.Parameter(torch.zeros(num_labels))
        nn.init.xavier_uniform_(self.label_weight)
        self.risk_global_classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, self.num_risk_factors),
        )
        self.protective_global_classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, 5),
        )





    def forward(
            self,
            token_hidden,
            attention_mask,
            document_repr
    ):
        batch, chunks, length, hidden = token_hidden.shape
        tokens = self.norm(token_hidden.reshape(batch, chunks * length, hidden))
        mask = attention_mask.reshape(batch, chunks * length).bool()
        # [B,T,K] -> normalise across T separately for every factor K.
        scores = torch.einsum("bth,kh->btk", tokens, self.label_queries) / math.sqrt(hidden)
        scores = scores.masked_fill(~mask.unsqueeze(-1), -1e4)
        weights = torch.softmax(scores, dim=1)
        label_repr = torch.einsum("btk,bth->bkh", weights, tokens)
        local_logits = (label_repr * self.label_weight.unsqueeze(0)).sum(-1) + self.label_bias
        global_logits = torch.cat([
            self.risk_global_classifier(document_repr),
            self.protective_global_classifier(document_repr),
        ], dim=-1)
        return local_logits + global_logits





# ============================================================
# Optional classifier with hidden layer
# ============================================================


class MLPHead(nn.Module):

    """
    General purpose head


    Used for experiments

    """



    def __init__(
            self,
            hidden_size,
            output_size
    ):


        super().__init__()



        self.net=nn.Sequential(

            nn.Linear(
                hidden_size,
                hidden_size//2
            ),

            nn.GELU(),

            nn.Dropout(
                0.2
            ),

            nn.Linear(
                hidden_size//2,
                output_size
            )

        )




    def forward(
            self,
            x
    ):

        return self.net(x)





# ============================================================
# Debug
# ============================================================


if __name__=="__main__":



    batch=2


    hidden=config.HIDDEN_SIZE



    doc_repr=torch.randn(

        batch,

        hidden

    )



    token_repr=torch.randn(

        batch,

        8,

        512,

        hidden

    )



    risk_head=RiskClassificationHead()



    evidence_head=EvidenceExtractionHead()



    factor_head=FactorClassificationHead()




    print(
        "Risk:",
        risk_head(
            doc_repr
        ).shape
    )



    s,e=evidence_head(
        token_repr
    )


    print(
        "Evidence:",
        s.shape,
        e.shape
    )



    print(
        "Factor:",
        factor_head(
            token_repr,
            torch.ones(batch, 8, 512, dtype=torch.long),
            doc_repr
        ).shape
    )
