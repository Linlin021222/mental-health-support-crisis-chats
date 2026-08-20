# ============================================================
# Suicide Risk Detection Competition
# Multi-task Model
# ============================================================


import torch

import torch.nn as nn



from configs.config import config



from models.backbone import (
    DebertaBackbone
)



from models.pooling import (
    DocumentPooling
)



from models.heads import (
    RiskClassificationHead,
    EvidenceExtractionHead,
    FactorClassificationHead
)





class SuicideRiskMultiTaskModel(nn.Module):

    """
    Full multitask model


    Tasks:

    1. Risk classification

    2. Evidence extraction

    3. Factor classification


    """



    def __init__(
            self
    ):


        super().__init__()



        # --------------------------------
        # Encoder
        # --------------------------------


        self.backbone=DebertaBackbone()



        # --------------------------------
        # Pooling
        # --------------------------------


        self.pooling=DocumentPooling()



        # --------------------------------
        # Task heads
        # --------------------------------


        self.risk_head=RiskClassificationHead()



        self.evidence_head=EvidenceExtractionHead()



        self.factor_head=FactorClassificationHead()





    def forward(
            self,
            input_ids,
            attention_mask
    ):


        """

        Args:


        input_ids:

        [B,C,L]


        attention_mask:

        [B,C,L]


        Returns:


        dict:

            risk_logits

            start_logits

            end_logits

            factor_logits


        """



        # ====================================================
        # Backbone
        # ====================================================


        token_hidden=self.backbone(

            input_ids,

            attention_mask

        )

        # The encoder can return fp16 under AMP while the custom pooling and
        # heads keep fp32 parameters.  Cast at this boundary to avoid the
        # ``mat1 Half / mat2 Float`` error on Windows CUDA builds.
        token_hidden=token_hidden.float()



        # token_hidden:

        # [B,C,L,H]




        # ====================================================
        # Document representation
        # ====================================================


        document_repr=self.pooling(

            token_hidden,

            attention_mask

        )



        # document_repr:

        # [B,H]




        # ====================================================
        # Three tasks
        # ====================================================


        risk_logits=self.risk_head(

            document_repr

        )



        start_logits,end_logits=self.evidence_head(

            token_hidden

        )



        factor_logits=self.factor_head(

            token_hidden,

            attention_mask,

            document_repr

        )




        return {


            "risk_logits":

                risk_logits,



            "start_logits":

                start_logits,



            "end_logits":

                end_logits,



            "factor_logits":

                factor_logits,



            # optional outputs

            "token_hidden":

                token_hidden,



            "document_repr":

                document_repr

        }





# ============================================================
# Freeze / unfreeze helper
# ============================================================


def freeze_encoder(
        model
):

    """
    Freeze DeBERTa

    """



    for p in model.backbone.parameters():

        p.requires_grad=False





def unfreeze_encoder(
        model
):

    """
    Enable DeBERTa training

    """



    for p in model.backbone.parameters():

        p.requires_grad=True





# ============================================================
# Parameter groups
# ============================================================


def get_optimizer_parameters(
        model
):

    """

    Different learning rates:

    backbone:

    1e-5


    heads:

    3e-5


    """



    backbone_params=[]

    head_params=[]



    for name,param in model.named_parameters():



        if not param.requires_grad:

            continue



        if "backbone" in name:


            backbone_params.append(
                param
            )


        else:


            head_params.append(
                param
            )



    return [

        {

            "params":
                backbone_params,

            "lr":
                config.BACKBONE_LR

        },


        {

            "params":
                head_params,

            "lr":
                config.HEAD_LR

        }

    ]





# ============================================================
# Debug
# ============================================================


if __name__=="__main__":



    model=SuicideRiskMultiTaskModel()



    input_ids=torch.randint(

        0,

        1000,

        (

            2,

            4,

            512

        )

    )



    mask=torch.ones_like(
        input_ids
    )



    output=model(

        input_ids,

        mask

    )



    for k,v in output.items():


        if torch.is_tensor(v):

            print(
                k,
                v.shape
            )
