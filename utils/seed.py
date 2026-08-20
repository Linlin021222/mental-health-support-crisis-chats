# ============================================================
# Suicide Risk Detection Competition
# Random Seed Utilities
# ============================================================


import os

import random

import numpy as np


import torch





def seed_everything(
        seed=42
):

    """

    Fix all random seeds


    Args:

        seed:

            random seed value


    """



    # -------------------------
    # Python
    # -------------------------


    random.seed(
        seed
    )



    # -------------------------
    # Environment
    # -------------------------


    os.environ[

        "PYTHONHASHSEED"

    ] = str(seed)





    # -------------------------
    # NumPy
    # -------------------------


    np.random.seed(
        seed
    )





    # -------------------------
    # PyTorch CPU
    # -------------------------


    torch.manual_seed(
        seed
    )





    # -------------------------
    # PyTorch GPU
    # -------------------------


    torch.cuda.manual_seed(
        seed
    )


    torch.cuda.manual_seed_all(
        seed
    )





    # -------------------------
    # CUDNN
    # -------------------------


    torch.backends.cudnn.deterministic=True


    torch.backends.cudnn.benchmark=False





def worker_init_fn(
        worker_id
):

    """

    DataLoader worker seed


    Important for:

    num_workers > 0


    """



    seed=torch.initial_seed()%2**32



    np.random.seed(
        seed
    )


    random.seed(
        seed
    )





def set_cuda():

    """

    Check CUDA environment


    """



    if torch.cuda.is_available():


        print(

            "CUDA available:",

            torch.cuda.get_device_name(0)

        )


    else:


        print(

            "CUDA not available"

        )





# ============================================================
# Debug
# ============================================================


if __name__=="__main__":


    seed_everything(
        42
    )


    set_cuda()



    print(
        np.random.rand(3)
    )


    print(
        torch.rand(3)
    )