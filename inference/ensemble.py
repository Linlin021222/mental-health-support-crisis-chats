# ============================================================
# Suicide Risk Detection Competition
# Model Ensemble
# ============================================================


import os

import torch

import numpy as np


from tqdm import tqdm


from torch.utils.data import DataLoader



from configs.config import config



from datasets.dataset import (
    SuicideRiskDataset
)



from datasets.collator import (
    SuicideRiskCollator
)



from models.multitask_model import (
    SuicideRiskMultiTaskModel
)





# ============================================================
# Load one model
# ============================================================


def load_model(
        checkpoint
):


    model=SuicideRiskMultiTaskModel()



    state=torch.load(

        checkpoint,

        map_location="cpu"

    )



    model.load_state_dict(
        state
    )



    model.cuda()


    model.eval()



    return model





# ============================================================
# Predict one model
# ============================================================


@torch.no_grad()

def predict_model(
        model,
        loader
):



    risk_logits=[]


    factor_logits=[]


    start_logits=[]


    end_logits=[]



    for batch in tqdm(

        loader,

        desc="Inference"

    ):



        input_ids=batch[

            "input_ids"

        ].cuda()



        attention_mask=batch[

            "attention_mask"

        ].cuda()



        outputs=model(

            input_ids,

            attention_mask

        )



        risk_logits.append(

            outputs[

                "risk_logits"

            ]

            .cpu()

            .numpy()

        )



        factor_logits.append(

            outputs[

                "factor_logits"

            ]

            .cpu()

            .numpy()

        )



        start_logits.append(

            outputs[

                "start_logits"

            ]

            .cpu()

            .numpy()

        )



        end_logits.append(

            outputs[

                "end_logits"

            ]

            .cpu()

            .numpy()

        )




    return {


        "risk":

            np.concatenate(

                risk_logits

            ),



        "factor":

            np.concatenate(

                factor_logits

            ),



        "start":

            np.concatenate(

                start_logits

            ),



        "end":

            np.concatenate(

                end_logits

            )

    }





# ============================================================
# Ensemble
# ============================================================


def ensemble_predict(
        checkpoints
):


    dataset=SuicideRiskDataset(

        os.path.join(

            config.CACHE_DIR,

            "test_cache.pt"

        )

    )



    loader=DataLoader(

        dataset,

        batch_size=config.BATCH_SIZE,

        shuffle=False,

        collate_fn=SuicideRiskCollator()

    )



    predictions=[]



    for ckpt in checkpoints:



        print(

            "Loading:",

            ckpt

        )



        model=load_model(
            ckpt
        )



        pred=predict_model(

            model,

            loader

        )



        predictions.append(
            pred
        )



        del model


        torch.cuda.empty_cache()





    # ========================================================
    # Average logits
    # ========================================================



    ensemble={}



    for key in [

        "risk",

        "factor",

        "start",

        "end"

    ]:



        ensemble[key]=np.mean(

            [

                p[key]

                for p in predictions

            ],

            axis=0

        )





    torch.save(

        ensemble,

        os.path.join(

            config.OUTPUT_DIR,

            "ensemble_logits.pt"

        )

    )



    print(

        "Ensemble finished"

    )



    return ensemble





# ============================================================
# Debug
# ============================================================


if __name__=="__main__":



    checkpoints=[


        "outputs/fold0/best_model.pt",


        "outputs/fold1/best_model.pt",


        "outputs/fold2/best_model.pt",


        "outputs/fold3/best_model.pt",


        "outputs/fold4/best_model.pt"

    ]



    ensemble_predict(

        checkpoints

    )