# ============================================================
# Suicide Risk Detection Competition
# Suicide Factor Processing
# ============================================================

import numpy as np

from configs.config import config



# ============================================================
# Factor vocabulary
# ============================================================

FACTOR_NAMES = config.FACTOR_LABELS



# ============================================================
# Encode factor labels
# ============================================================

def encode_factors(
        factors
):
    """
    Convert factor names into multi-hot vector


    Input:

    [
      "hopelessness",
      "poor social support"
    ]


    Output:

    [
      0,
      0,
      0,
      1,
      ...
    ]

    """



    vector=np.zeros(
        config.NUM_FACTORS,
        dtype=np.float32
    )


    if factors is None:
        return vector



    if not isinstance(
        factors,
        list
    ):

        factors=[
            factors
        ]



    for factor in factors:


        factor=str(
            factor
        ).strip()



        if factor in config.FACTOR2ID:


            idx=config.FACTOR2ID[
                factor
            ]

            vector[idx]=1



    return vector





# ============================================================
# Decode prediction
# ============================================================

def decode_factors(
        probabilities,
        threshold=0.5
):

    """
    Convert sigmoid probabilities
    to factor labels


    Example:

    probs:

    [
    0.1,
    0.8,
    0.7
    ]


    output:

    [
    "hopelessness",
    "emotion dysregulation"
    ]

    """



    labels=[]



    for idx,p in enumerate(
        probabilities
    ):


        if p >= threshold:


            labels.append(
                config.ID2FACTOR[idx]
            )



    return labels




# ============================================================
# Dynamic threshold search
# ============================================================

def search_best_threshold(
        y_true,
        y_prob,
        metric_func,
        thresholds=None
):

    """
    Search threshold maximizing Macro F1


    Because factor distribution
    is highly imbalanced.


    """



    if thresholds is None:

        thresholds=np.arange(
            0.05,
            0.95,
            0.05
        )



    best_score=0

    best_threshold=0.5



    for t in thresholds:


        y_pred=(
            y_prob>=t
        ).astype(
            int
        )


        score=metric_func(
            y_true,
            y_pred
        )



        if score>best_score:

            best_score=score

            best_threshold=t



    return (
        best_threshold,
        best_score
    )





# ============================================================
# Factor statistics
# ============================================================

def factor_distribution(
        dataframe
):

    """
    Calculate factor frequency


    Used for:
        - class weight
        - analysis
        - threshold initialization

    """



    counts=np.zeros(
        config.NUM_FACTORS
    )


    for vector in dataframe[
        "factor_vector"
    ]:


        counts += vector



    return {
        config.ID2FACTOR[i]:
        int(counts[i])

        for i in range(
            config.NUM_FACTORS
        )
    }





# ============================================================
# Positive weight calculation
# ============================================================

def calculate_pos_weight(
        dataframe
):

    """
    Calculate BCE positive weights


    Formula:

    neg / pos


    """



    labels=np.stack(
        dataframe[
            "factor_vector"
        ].values
    )



    positive=labels.sum(
        axis=0
    )


    negative=(
        len(labels)
        -
        positive
    )



    weights=(
        negative
        /
        (
            positive
            +1e-6
        )
    )


    return weights.astype(
        np.float32
    )





# ============================================================
# Factor evaluation helper
# ============================================================

def multilabel_to_text(
        vector
):

    """
    Convert multi-hot vector
    back to readable labels

    """

    result=[]


    for idx,value in enumerate(
        vector
    ):


        if value==1:

            result.append(
                config.ID2FACTOR[idx]
            )



    return result