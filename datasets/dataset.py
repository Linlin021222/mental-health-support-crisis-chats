# ============================================================
# Suicide Risk Detection Competition
# PyTorch Dataset
# ============================================================


import torch

from torch.utils.data import Dataset



from configs.config import config





class SuicideRiskDataset(Dataset):

    """
    Dataset for multitask suicide risk model


    Each item contains:


    input_ids
    attention_mask

    risk_label

    start_labels
    end_labels

    factor_vector


    """



    def __init__(
            self,
            cache_file
    ):


        super().__init__()



        self.data=torch.load(
            cache_file,
            map_location="cpu"
        )



    def __len__(
            self
    ):


        return len(
            self.data
        )



    def __getitem__(
            self,
            index
    ):


        item=self.data[index]



        sample={



            # --------------------
            # basic information
            # --------------------

            "row_id":
                item["row_id"],

            "text": item.get("text", ""),

            "offset_mapping": item.get("offset_mapping"),

            "evidence": item.get("evidence", []),



            # --------------------
            # encoder input
            # --------------------

            "input_ids":
                item["input_ids"],



            "attention_mask":
                item["attention_mask"],




            # --------------------
            # risk classification
            # --------------------

            "risk_label":
                item["risk_label"],




            # --------------------
            # evidence extraction
            # --------------------

            "start_labels":
                item["start_labels"],



            "end_labels":
                item["end_labels"],

            "token_labels":
                item.get("token_labels", torch.zeros_like(item["start_labels"])),




            # --------------------
            # factor prediction
            # --------------------

            "factor_vector":
                item["factor_vector"]

        }



        return sample





# ============================================================
# Train / validation split
# ============================================================


def split_dataset(
        dataset,
        train_indices,
        valid_indices
):

    """
    Create train and validation subsets

    Used by K-fold CV


    """



    train_set=torch.utils.data.Subset(
        dataset,
        train_indices
    )



    valid_set=torch.utils.data.Subset(
        dataset,
        valid_indices
    )



    return (
        train_set,
        valid_set
    )





# ============================================================
# Dataset statistics
# ============================================================

def inspect_dataset(
        dataset,
        n=3
):

    """
    Debug dataset content
    """



    for i in range(
        min(
            n,
            len(dataset)
        )
    ):


        item=dataset[i]


        print(
            "="*50
        )


        print(
            "row_id:",
            item["row_id"]
        )


        print(
            "input shape:",
            item["input_ids"].shape
        )


        print(
            "risk:",
            item["risk_label"]
        )


        print(
            "factor:",
            item["factor_vector"]
        )





# ============================================================
# Debug
# ============================================================


if __name__=="__main__":


    dataset=SuicideRiskDataset(

        config.CACHE_DIR
        +
        "/train_cache.pt"

    )



    print(
        "Dataset size:",
        len(dataset)
    )



    inspect_dataset(
        dataset
    )
