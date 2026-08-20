# ============================================================
# Suicide Risk Detection Competition
# Backbone Encoder
# ============================================================


import torch

import torch.nn as nn


from transformers import AutoModel


from configs.config import config





class DebertaBackbone(nn.Module):

    """
    DeBERTa backbone with chunk support


    Input:

    input_ids:

    [batch, chunks, seq_len]


    attention_mask:

    [batch, chunks, seq_len]



    Output:

    hidden states:


    [batch, chunks, seq_len, hidden]

    """



    def __init__(
            self,
            model_name=config.MODEL_NAME
    ):


        super().__init__()



        self.encoder=AutoModel.from_pretrained(
            model_name,
            # The checkpoint was already downloaded successfully. Avoid a
            # network metadata request on every local run.
            local_files_only=True,
            # GradScaler requires trainable master parameters in FP32. AMP
            # still executes eligible forward operations in FP16.
            dtype=torch.float32
        )

        # Reduce activation memory enough for DeBERTa-base on an 8 GB laptop
        # GPU. This trades some speed for substantially lower VRAM use.
        self.encoder.gradient_checkpointing_enable()
        self.encoder.config.use_cache=False



        self.hidden_size=(
            self.encoder.config.hidden_size
        )



    def forward(
            self,
            input_ids,
            attention_mask
    ):


        batch_size=input_ids.size(0)


        num_chunks=input_ids.size(1)


        seq_len=input_ids.size(2)



        # ----------------------------------------------------
        # merge chunk dimension
        # ----------------------------------------------------


        input_ids=input_ids.reshape(

            batch_size*num_chunks,

            seq_len

        )


        attention_mask=attention_mask.reshape(

            batch_size*num_chunks,

            seq_len

        )




        # ----------------------------------------------------
        # transformer forward
        # ----------------------------------------------------


        outputs=self.encoder(

            input_ids=input_ids,

            attention_mask=attention_mask

        )



        hidden_states=outputs.last_hidden_state



        # shape:

        # [
        # batch*chunks,
        # seq_len,
        # hidden
        # ]




        # ----------------------------------------------------
        # restore chunk dimension
        # ----------------------------------------------------


        hidden_states=hidden_states.reshape(

            batch_size,

            num_chunks,

            seq_len,

            self.hidden_size

        )



        return hidden_states





# ============================================================
# Freeze utilities
# ============================================================


def freeze_backbone(
        model,
        freeze=True
):

    """
    Freeze transformer parameters


    Useful for:
    first epoch warmup


    """



    for param in model.encoder.parameters():

        param.requires_grad=not freeze





def unfreeze_last_layers(
        model,
        num_layers=4
):

    """
    Unfreeze last N transformer layers


    Used for gradual fine tuning


    """



    layers=model.encoder.encoder.layer



    total=len(
        layers
    )



    for i,layer in enumerate(
        layers
    ):


        if i >= total-num_layers:


            for param in layer.parameters():

                param.requires_grad=True





# ============================================================
# Parameter count
# ============================================================


def count_parameters(
        model
):

    """

    Count trainable parameters

    """



    return sum(

        p.numel()

        for p in model.parameters()

        if p.requires_grad

    )





# ============================================================
# Debug
# ============================================================


if __name__=="__main__":


    model=DebertaBackbone()



    print(
        "Hidden size:",
        model.hidden_size
    )


    dummy_ids=torch.randint(

        0,

        1000,

        (

            2,

            4,

            512

        )

    )


    dummy_mask=torch.ones_like(
        dummy_ids
    )



    out=model(

        dummy_ids,

        dummy_mask

    )



    print(
        out.shape
    )
