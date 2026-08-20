# ============================================================
# Suicide Risk Detection Competition
# Evaluation Metrics
# ============================================================


import numpy as np
from utils.task1_metric import task1_score, composite_score


from sklearn.metrics import (

    f1_score,

    precision_score,

    recall_score

)





# ============================================================
# Risk Weighted F1
# ============================================================


def risk_weighted_f1(
        y_true,
        y_pred
):

    """
    Subtask 1:

    Suicide risk classification


    Metric:

    Weighted F1


    """



    return f1_score(

        y_true,

        y_pred,

        average="weighted"

    )





# ============================================================
# Evidence span matching
# ============================================================


def normalize_text(
        text
):


    return (

        text

        .lower()

        .strip()

    )





def span_match(
        pred,
        gold
):

    """

    Check whether evidence phrases match


    Rule:

    pred contains gold

    OR

    gold contains pred



    """



    pred=normalize_text(
        pred
    )


    gold=normalize_text(
        gold
    )



    return (

        pred in gold

        or

        gold in pred

    )





def evidence_phrase_f1(
        predictions,
        ground_truths
):
    """Official per-post Phrase F1 with maximum one-to-one matching."""
    post_scores=[]
    for preds,golds in zip(predictions, ground_truths):
        if not preds and not golds:
            post_scores.append(1.0)
            continue
        edges=[]
        for pred in preds:
            pred_normal=normalize_text(pred)
            matches=[]
            for gold_index,gold in enumerate(golds):
                gold_normal=normalize_text(gold)
                contained=(pred_normal in gold_normal or gold_normal in pred_normal)
                length_ok=(
                    len(pred_normal.split())
                    <= 3*max(1, len(gold_normal.split()))
                )
                if pred_normal and gold_normal and contained and length_ok:
                    matches.append(gold_index)
            edges.append(matches)

        matched={}

        def augment(pred_index, visited):
            for gold_index in edges[pred_index]:
                if gold_index in visited:
                    continue
                visited.add(gold_index)
                if gold_index not in matched or augment(matched[gold_index], visited):
                    matched[gold_index]=pred_index
                    return True
            return False

        true_positive=sum(augment(i, set()) for i in range(len(preds)))
        precision=true_positive/len(preds) if preds else 0.0
        recall=true_positive/len(golds) if golds else 0.0
        post_scores.append(
            0.0 if precision+recall==0
            else 2*precision*recall/(precision+recall)
        )
    return float(np.mean(post_scores)) if post_scores else 0.0





# ============================================================
# Factor Macro F1
# ============================================================


def factor_macro_f1(
        y_true,
        y_pred
):

    """

    Subtask 2:

    Multi-label factor classification


    """



    return f1_score(

        y_true,

        y_pred,

        average="macro"

    )





# ============================================================
# Threshold prediction
# ============================================================


def sigmoid(
        x
):


    return 1/(

        1+

        np.exp(-x)

    )





def factor_threshold(
        logits,
        threshold=0.5
):

    """

    Convert logits to labels

    """



    probs=sigmoid(
        logits
    )


    return (

        probs >= threshold

    ).astype(
        int
    )





# ============================================================
# Competition final score
# ============================================================


def final_score(
        risk_f1,
        evidence_f1,
        factor_f1
):

    """

    Overall score:


    70% Subtask1

    30% Subtask2



    Subtask1:

    risk + evidence


    """



    return composite_score(risk_f1, evidence_f1, factor_f1)





# ============================================================
# Debug
# ============================================================


if __name__=="__main__":



    print(

        "Risk F1:",

        risk_weighted_f1(

            [0,1,2,3],

            [0,1,1,3]

        )

    )



    print(

        "Phrase F1:",

        evidence_phrase_f1(

            [

                ["kill myself"],

                ["want to die"]

            ],

            [

                ["I want to kill myself"],

                ["want to die"]

            ]

        )

    )



    print(

        "Factor F1:",

        factor_macro_f1(

            [

                [1,0,1],

                [0,1,0]

            ],

            [

                [1,0,0],

                [0,1,0]

            ]

        )

    )
