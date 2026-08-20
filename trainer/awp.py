# ============================================================
# Suicide Risk Detection Competition
# Adversarial Weight Perturbation
# ============================================================


import torch





class AWP:

    """
    Adversarial Weight Perturbation


    Reference:

    Improving Neural Network Generalization
    by Adversarial Weight Perturbation


    """



    def __init__(
            self,
            model,
            optimizer,
            adv_lr=1e-3,
            adv_eps=1e-3,
            start_epoch=0
    ):


        self.model=model


        self.optimizer=optimizer


        self.adv_lr=adv_lr


        self.adv_eps=adv_eps


        self.start_epoch=start_epoch



        self.backup={}


        self.backup_eps={}





    # ========================================================
    # Save original weights
    # ========================================================


    def save(
            self
    ):


        """

        Backup parameters before attack


        """



        for name,param in self.model.named_parameters():


            if param.requires_grad:


                self.backup[name]=param.data.clone()





    # ========================================================
    # Attack
    # ========================================================


    def attack(
            self,
            epoch
    ):


        """

        Add adversarial perturbation


        """



        if epoch < self.start_epoch:

            return




        for name,param in self.model.named_parameters():


            if param.requires_grad and param.grad is not None:


                self.backup_eps[name]=(

                    self.adv_eps
                    *
                    param.abs().detach()

                )



                norm1=torch.norm(
                    param.grad
                )


                norm2=torch.norm(
                    param.data
                )



                if norm1 != 0 and norm2 !=0:


                    r_at=(

                        self.adv_lr
                        *
                        param.grad
                        /
                        norm1

                    )



                    param.data.add_(
                        r_at
                    )



                    param.data=torch.max(

                        torch.min(

                            param.data,

                            self.backup[name]
                            +
                            self.backup_eps[name]

                        ),

                        self.backup[name]
                        -
                        self.backup_eps[name]

                    )





    # ========================================================
    # Restore
    # ========================================================


    def restore(
            self
    ):


        """

        Restore original parameters

        """



        for name,param in self.model.named_parameters():


            if name in self.backup:


                param.data=self.backup[name]



        self.backup={}


        self.backup_eps={}





# ============================================================
# Context manager
# ============================================================


class AWPTrainer:


    """

    Helper wrapper


    Usage:


    awp.attack()

    loss.backward()

    awp.restore()


    """



    def __init__(
            self,
            awp
    ):


        self.awp=awp





    def __enter__(
            self
    ):


        self.awp.save()


        return self.awp





    def __exit__(
            self,
            exc_type,
            exc_val,
            exc_tb
    ):


        self.awp.restore()





# ============================================================
# Debug
# ============================================================


if __name__=="__main__":


    import torch.nn as nn


    model=nn.Linear(
        10,
        2
    )


    optimizer=torch.optim.Adam(
        model.parameters()
    )


    awp=AWP(

        model,

        optimizer

    )


    x=torch.randn(
        4,
        10
    )


    loss=model(x).sum()


    loss.backward()


    awp.save()


    awp.attack(
        epoch=1
    )


    awp.restore()


    print(
        "AWP finished"
    )