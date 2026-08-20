# ============================================================
# Suicide Risk Detection Competition
# Evidence Extraction Processing
# ============================================================

import re
import numpy as np



# ============================================================
# Normalize text
# ============================================================

def normalize_text(text):

    """
    Used for matching evidence phrase

    Case insensitive normalization
    """

    if text is None:
        return ""


    text=text.lower()


    # normalize whitespace

    text=re.sub(
        r"\s+",
        " ",
        text
    )


    return text.strip()



# ============================================================
# Find phrase position
# ============================================================

def find_phrase_position(
        text,
        phrase
):
    """
    Find evidence phrase position
    in original text


    Return:

    [
        (start,end),
        ...
    ]

    """

    positions=[]


    if not phrase:
        return positions



    text_norm=normalize_text(
        text
    )

    phrase_norm=normalize_text(
        phrase
    )


    start=0


    while True:


        idx=text_norm.find(
            phrase_norm,
            start
        )


        if idx==-1:
            break



        end=idx+len(
            phrase_norm
        )


        positions.append(
            (
                idx,
                end
            )
        )


        start=end



    return positions




# ============================================================
# Locate all evidence phrases
# ============================================================

def locate_evidence_spans(
        text,
        evidence_list
):

    """
    Convert:

    [
     "kill myself",
     "hope I die"
    ]


    into character spans


    """

    spans=[]


    for phrase in evidence_list:


        matches=find_phrase_position(
            text,
            phrase
        )


        for m in matches:

            spans.append(m)



    return spans




# ============================================================
# Character span -> token span
# ============================================================

def char_to_token_span(
        offsets,
        char_start,
        char_end
):
    """
    Convert character positions
    to tokenizer token positions.


    offsets example:

    [
      (0,5),
      (6,10)
    ]


    """

    token_start=None

    token_end=None



    for idx,(s,e) in enumerate(offsets):


        # skip special tokens

        if s==0 and e==0:
            continue



        # token overlaps evidence

        if e>char_start and s<char_end:


            if token_start is None:

                token_start=idx


            token_end=idx



    return token_start,token_end





# ============================================================
# Build BIO-like span labels
# ============================================================

def build_evidence_labels(
        text,
        evidence,
        offsets
):

    """
    Generate start/end labels


    Example:


    start_labels:

    [0,0,1,0,0]


    end_labels:

    [0,0,0,1,0]


    """



    length=len(offsets)


    start_labels=np.zeros(
        length,
        dtype=np.float32
    )


    end_labels=np.zeros(
        length,
        dtype=np.float32
    )



    spans=locate_evidence_spans(
        text,
        evidence
    )



    for char_start,char_end in spans:


        token_start,token_end=char_to_token_span(
            offsets,
            char_start,
            char_end
        )


        if token_start is None:
            continue



        start_labels[
            token_start
        ]=1


        end_labels[
            token_end
        ]=1



    return (
        start_labels,
        end_labels
    )





# ============================================================
# Decode prediction spans
# ============================================================

def decode_spans(
        text,
        offsets,
        start_prob,
        end_prob,
        start_threshold=0.35,
        end_threshold=0.35,
        max_length=64
):

    """
    Convert model prediction
    into evidence phrases


    """



    candidates=[]


    starts=np.where(
        start_prob > start_threshold
    )[0]


    ends=np.where(
        end_prob > end_threshold
    )[0]



    for s in starts:


        possible=[]


        for e in ends:


            if e < s:
                continue


            if e-s+1 > max_length:
                continue


            possible.append(e)



        if len(possible)==0:
            continue



        e=min(
            possible
        )


        candidates.append(
            (
                s,
                e
            )
        )



    results=[]


    for s,e in candidates:


        char_start=offsets[s][0]

        char_end=offsets[e][1]


        phrase=text[
            char_start:
            char_end
        ].strip()



        if phrase:

            results.append(
                phrase
            )


    return results




# ============================================================
# Phrase F1 matching
# ============================================================

def phrase_match(
        pred,
        gold
):

    """
    Official matching rule:

    after lowercase normalization:

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



def calculate_phrase_f1(
        predictions,
        targets
):

    """
    predictions:

    [
      ["kill myself"]
    ]


    targets:

    [
      ["want to kill myself"]
    ]

    """


    tp=0

    fp=0

    fn=0



    for pred,gold in zip(
        predictions,
        targets
    ):


        matched_gold=set()


        for p in pred:


            found=False


            for i,g in enumerate(gold):


                if i in matched_gold:
                    continue



                if phrase_match(
                    p,
                    g
                ):


                    tp+=1

                    matched_gold.add(i)

                    found=True

                    break



            if not found:

                fp+=1



        fn += (
            len(gold)
            -
            len(matched_gold)
        )



    precision = (
        tp/(tp+fp)
        if tp+fp>0
        else 0
    )


    recall = (
        tp/(tp+fn)
        if tp+fn>0
        else 0
    )


    if precision+recall==0:

        return 0



    return (
        2*precision*recall
        /
        (precision+recall)
    )