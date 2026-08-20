# ============================================================
# Suicide Risk Detection Competition
# Data Collator
# ============================================================


import torch


from configs.config import config





class SuicideRiskCollator:

    """
    Collate function for multitask model


    Input:

    [
      sample1,
      sample2,
      ...
    ]


    Output:

    batch dictionary


    """



    def __init__(
            self
    ):

        pass





    def __call__(
            self,
            batch
    ):


        # ====================================================
        # Basic information
        # ====================================================


        row_ids=[
            x["row_id"]
            for x in batch
        ]

        texts=[x.get("text", "") for x in batch]
        offset_mappings=[x.get("offset_mapping") for x in batch]
        evidences=[x.get("evidence", []) for x in batch]




        # ====================================================
        # Encoder inputs
        # ====================================================


        input_ids=torch.stack(
            [
                x["input_ids"]
                for x in batch
            ]
        )


        attention_mask=torch.stack(
            [
                x["attention_mask"]
                for x in batch
            ]
        )



        # shape:

        # input_ids:
        #
        # [batch,
        #  chunks,
        #  seq_len]




        # ====================================================
        # Risk label
        # ====================================================


        risk_labels=torch.stack(
            [
                x["risk_label"]
                for x in batch
            ]
        )





        # ====================================================
        # Evidence labels
        # ====================================================


        start_labels=torch.stack(
            [
                x["start_labels"]
                for x in batch
            ]
        )



        end_labels=torch.stack(
            [
                x["end_labels"]
                for x in batch
            ]
        )

        token_labels=torch.stack(
            [x.get("token_labels", torch.zeros_like(x["start_labels"])) for x in batch]
        )




        # ====================================================
        # Factor labels
        # ====================================================


        factor_vectors=torch.stack(
            [
                x["factor_vector"]
                for x in batch
            ]
        )





        output={


            "row_id":
                row_ids,

            "texts": texts,

            "offset_mappings": offset_mappings,

            "evidences": evidences,


            "input_ids":
                input_ids,


            "attention_mask":
                attention_mask,


            "risk_labels":
                risk_labels,


            "start_labels":
                start_labels,


            "end_labels":
                end_labels,

            "token_labels":
                token_labels,


            "factor_vectors":
                factor_vectors

        }



        return output





# ============================================================
# Test collator
# ============================================================


if __name__=="__main__":


    from torch.utils.data import DataLoader


    from datasets.dataset import (
        SuicideRiskDataset
    )


    dataset=SuicideRiskDataset(

        config.CACHE_DIR
        +
        "/train_cache.pt"

    )



    loader=DataLoader(

        dataset,

        batch_size=2,

        shuffle=True,

        collate_fn=SuicideRiskCollator()

    )



    batch=next(
        iter(loader)
    )



    for k,v in batch.items():

        if torch.is_tensor(v):

            print(
                k,
                v.shape
            )

        else:

            print(
                k,
                type(v)
            )
