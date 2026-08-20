# ============================================================
# Suicide Risk Detection Competition
# Exponential Moving Average
# ============================================================


import torch





class EMA:

    """
    Exponential Moving Average


    Maintain shadow parameters:


    shadow =
        decay * shadow
        +
        (1-decay) * parameter


    """



    def __init__(
            self,
            model,
            decay=0.999
    ):


        self.model=model


        self.decay=decay


        self.shadow={}


        self.backup={}



    # ========================================================
    # Initialize
    # ========================================================


    def register(
            self
    ):


        """
        Save initial parameters

        """

        for name,param in self.model.named_parameters():


            if param.requires_grad:


                self.shadow[name]=param.data.clone()





    # ========================================================
    # Update EMA
    # ========================================================


    def update(
            self
    ):


        """
        Update moving average

        Call after optimizer.step()

        """



        for name,param in self.model.named_parameters():


            if param.requires_grad:


                assert name in self.shadow



                new_average=(

                    self.decay
                    *
                    self.shadow[name]

                    +

                    (1-self.decay)
                    *
                    param.data

                )



                self.shadow[name]=new_average.clone()





    # ========================================================
    # Apply EMA weights
    # ========================================================


    def apply_shadow(
            self
    ):


        """
        Replace model weights
        with EMA weights


        Used before validation

        """



        for name,param in self.model.named_parameters():


            if param.requires_grad:


                self.backup[name]=param.data.clone()


                param.data=self.shadow[name]





    # ========================================================
    # Restore original weights
    # ========================================================


    def restore(
            self
    ):


        """
        Restore training weights

        """



        for name,param in self.model.named_parameters():


            if param.requires_grad:


                param.data=self.backup[name]



        self.backup={}





    # ========================================================
    # State dict
    # ========================================================


    def state_dict(
            self
    ):


        return {

            "shadow":
                self.shadow

        }



    def load_state_dict(
            self,
            state_dict
    ):


        self.shadow=state_dict["shadow"]





# ============================================================
# Debug
# ============================================================


if __name__=="__main__":


    import torch.nn as nn


    model=nn.Linear(
        10,
        2
    )


    ema=EMA(
        model,
        decay=0.999
    )


    ema.register()


    optimizer=torch.optim.Adam(
        model.parameters()
    )


    x=torch.randn(
        4,
        10
    )


    y=model(x).sum()


    y.backward()


    optimizer.step()


    ema.update()


    print(
        "EMA updated"
    )