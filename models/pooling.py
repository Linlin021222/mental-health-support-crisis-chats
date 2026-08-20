# ============================================================
# Suicide Risk Detection Competition
# Pooling Layers
# ============================================================


import torch

import torch.nn as nn

import torch.nn.functional as F



from configs.config import config





# ============================================================
# Token Attention Pooling
# ============================================================


class TokenAttentionPooling(nn.Module):

    """
    Pool tokens inside each chunk


    Input:

    hidden:

    [B,C,L,H]


    Output:

    chunk_repr:

    [B,C,H]

    """



    def __init__(
            self,
            hidden_size=config.HIDDEN_SIZE
    ):


        super().__init__()



        self.attention=nn.Sequential(

            nn.Linear(
                hidden_size,
                hidden_size
            ),

            nn.Tanh(),

            nn.Linear(
                hidden_size,
                1
            )

        )




    def forward(
            self,
            hidden_states,
            attention_mask
    ):


        """
        hidden_states:

        [B,C,L,H]


        attention_mask:

        [B,C,L]


        """



        scores=self.attention(
            hidden_states
        )



        # [B,C,L,1]



        scores=scores.squeeze(-1)



        # mask padding tokens

        scores=scores.masked_fill(

            attention_mask==0,

            # -1e9 overflows under fp16 AMP.  -1e4 is already effectively
            # zero after softmax and is representable by float16.
            -1e4

        )



        weights=torch.softmax(

            scores,

            dim=-1

        )



        weights=weights.unsqueeze(-1)



        chunk_repr=(

            hidden_states
            *
            weights

        ).sum(
            dim=2
        )



        return chunk_repr





# ============================================================
# Chunk Attention Pooling
# ============================================================


class ChunkAttentionPooling(nn.Module):

    """
    Pool multiple chunks


    Input:

    chunk_repr:

    [B,C,H]


    Output:

    document representation:

    [B,H]

    """



    def __init__(
            self,
            hidden_size=config.HIDDEN_SIZE
    ):


        super().__init__()



        self.attention=nn.Sequential(

            nn.Linear(
                hidden_size,
                hidden_size//2
            ),

            nn.Tanh(),

            nn.Linear(
                hidden_size//2,
                1
            )

        )




    def forward(
            self,
            chunk_repr,
            chunk_mask=None
    ):



        scores=self.attention(
            chunk_repr
        )



        scores=scores.squeeze(-1)



        if chunk_mask is not None:


            scores=scores.masked_fill(

                chunk_mask==0,

                # See token-level masking above: keep the mask fp16-safe.
                -1e4

            )



        weights=torch.softmax(

            scores,

            dim=1

        )



        weights=weights.unsqueeze(-1)



        doc_repr=(

            chunk_repr
            *
            weights

        ).sum(
            dim=1
        )



        return doc_repr





# ============================================================
# Mean Pooling
# ============================================================


class MeanPooling(nn.Module):

    """
    Simple masked mean pooling


    """



    def forward(
            self,
            hidden_states,
            attention_mask
    ):


        mask=attention_mask.unsqueeze(-1)



        masked_hidden=(

            hidden_states
            *
            mask

        )



        summed=masked_hidden.sum(
            dim=2
        )


        counts=mask.sum(
            dim=2
        ).clamp(
            min=1e-6
        )


        return summed/counts





# ============================================================
# CLS Pooling
# ============================================================


class CLSPooling(nn.Module):

    """
    Use CLS token representation

    """



    def forward(
            self,
            hidden_states
    ):


        return hidden_states[:,:,0]





# ============================================================
# Full document pooling
# ============================================================


class DocumentPooling(nn.Module):

    """
    Complete pooling module


    hidden:

    [B,C,L,H]


    output:

    [B,H]

    """



    def __init__(self):


        super().__init__()



        self.token_pool=TokenAttentionPooling()



        self.chunk_pool=ChunkAttentionPooling()




    def forward(
            self,
            hidden_states,
            attention_mask
    ):



        # -----------------------------
        # token -> chunk
        # -----------------------------


        chunk_repr=self.token_pool(

            hidden_states,

            attention_mask

        )



        # -----------------------------
        # chunk mask
        # -----------------------------


        chunk_mask=(

            attention_mask.sum(
                dim=-1
            )
            >
            0

        ).long()



        # -----------------------------
        # chunk -> document
        # -----------------------------


        doc_repr=self.chunk_pool(

            chunk_repr,

            chunk_mask

        )



        return doc_repr





# ============================================================
# Debug
# ============================================================


if __name__=="__main__":



    pooling=DocumentPooling()



    hidden=torch.randn(

        2,

        8,

        512,

        1024

    )



    mask=torch.ones(

        2,

        8,

        512

    )



    output=pooling(

        hidden,

        mask

    )



    print(
        output.shape
    )
