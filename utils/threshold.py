# ============================================================
# Suicide Risk Detection Competition
# Threshold Optimization
# ============================================================


import numpy as np


from sklearn.metrics import f1_score





# ============================================================
# Sigmoid
# ============================================================


def sigmoid(
        x
):


    return 1/(

        1+

        np.exp(-x)

    )





# ============================================================
# Single label threshold search
# ============================================================


def search_best_threshold(
        y_true,
        logits,
        thresholds=None
):

    """

    Search best threshold for one label


    Args:


    y_true:

        [N]


    logits:

        [N]


    """



    if thresholds is None:


        thresholds=np.arange(

            0.05,

            0.95,

            0.05

        )



    probs=sigmoid(
        logits
    )



    best_score=0


    best_threshold=0.5



    for t in thresholds:



        preds=(

            probs >= t

        ).astype(
            int
        )



        score=f1_score(

            y_true,

            preds,

            zero_division=0

        )



        if score > best_score:


            best_score=score


            best_threshold=t




    return (

        best_threshold,

        best_score

    )





# ============================================================
# Multi-label threshold search
# ============================================================


def optimize_factor_thresholds(
        y_true,
        logits
):

    """

    Optimize threshold for every factor


    Args:


    y_true:

        [N,24]


    logits:

        [N,24]



    Returns:


    thresholds:

        [24]


    """



    num_labels=y_true.shape[1]



    thresholds=[]



    scores=[]



    for i in range(
        num_labels
    ):



        t,score=search_best_threshold(

            y_true[:,i],

            logits[:,i]

        )



        thresholds.append(
            t
        )


        scores.append(
            score
        )



    return (

        np.array(
            thresholds
        ),

        np.array(
            scores
        )

    )





# ============================================================
# Apply thresholds
# ============================================================


def apply_thresholds(
        logits,
        thresholds
):

    """

    Convert logits into labels


    """



    probs=sigmoid(
        logits
    )



    return (

        probs >= thresholds

    ).astype(
        int
    )





# ============================================================
# Global threshold search
# ============================================================


def search_global_threshold(
        y_true,
        logits
):

    """

    Search one threshold for all labels


    Useful when validation data is small


    """



    thresholds=np.arange(

        0.05,

        0.95,

        0.05

    )



    probs=sigmoid(
        logits
    )


    best_t=0.5


    best_score=0



    for t in thresholds:


        pred=(

            probs>=t

        ).astype(
            int
        )



        score=f1_score(

            y_true,

            pred,

            average="macro",

            zero_division=0

        )



        if score>best_score:


            best_score=score


            best_t=t





    return (

        best_t,

        best_score

    )





# ============================================================
# Debug
# ============================================================


if __name__=="__main__":


    y_true=np.array(

        [

            [1,0,1],

            [0,1,0],

            [1,1,0],

            [0,0,1]

        ]

    )



    logits=np.random.randn(

        4,

        3

    )



    thresholds,scores=optimize_factor_thresholds(

        y_true,

        logits

    )



    print(
        "Thresholds:"
    )


    print(
        thresholds
    )


    print(
        "F1:"
    )


    print(
        scores
    )