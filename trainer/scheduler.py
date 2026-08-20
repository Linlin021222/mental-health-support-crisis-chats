# ============================================================
# Suicide Risk Detection Competition
# Learning Rate Scheduler
# ============================================================


import math


from torch.optim.lr_scheduler import LambdaLR





# ============================================================
# Linear Warmup + Cosine Decay
# ============================================================


def get_cosine_schedule_with_warmup(
        optimizer,
        num_training_steps,
        num_warmup_steps,
        num_cycles=0.5
):

    """
    Create scheduler:

    warmup:

        lr increases linearly


    decay:

        cosine decay



    """



    def lr_lambda(
            current_step
    ):


        # -------------------------
        # Warmup stage
        # -------------------------


        if current_step < num_warmup_steps:


            return float(current_step) / max(

                1,

                num_warmup_steps

            )



        # -------------------------
        # Cosine decay
        # -------------------------



        progress=(

            float(current_step-num_warmup_steps)

            /

            max(

                1,

                num_training_steps-num_warmup_steps

            )

        )



        return max(

            0.0,

            0.5 *

            (

                1.0

                +

                math.cos(

                    math.pi

                    *

                    float(num_cycles)

                    *

                    2.0

                    *

                    progress

                )

            )

        )



    return LambdaLR(

        optimizer,

        lr_lambda

    )





# ============================================================
# Scheduler builder
# ============================================================


def build_scheduler(
        optimizer,
        epochs,
        steps_per_epoch,
        warmup_ratio=0.1
):

    """

    Build scheduler automatically


    """



    total_steps=(

        epochs

        *

        steps_per_epoch

    )



    warmup_steps=int(

        total_steps

        *

        warmup_ratio

    )



    scheduler=get_cosine_schedule_with_warmup(

        optimizer,

        num_training_steps=total_steps,

        num_warmup_steps=warmup_steps

    )


    return scheduler





# ============================================================
# Learning rate monitor
# ============================================================


def get_lr(
        optimizer
):

    """

    Get current learning rate

    """



    return [

        group["lr"]

        for group in optimizer.param_groups

    ]





# ============================================================
# Debug
# ============================================================


if __name__=="__main__":


    import torch


    model=torch.nn.Linear(
        10,
        2
    )


    optimizer=torch.optim.AdamW(

        model.parameters(),

        lr=2e-5

    )



    scheduler=build_scheduler(

        optimizer,

        epochs=5,

        steps_per_epoch=100

    )



    for step in range(10):


        optimizer.step()


        scheduler.step()


        print(

            step,

            get_lr(
                optimizer
            )

        )