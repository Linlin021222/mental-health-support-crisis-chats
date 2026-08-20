# ============================================================
# Suicide Risk Detection Competition
# Tokenizer Utilities
# ============================================================

import numpy as np

from transformers import AutoTokenizer

from configs.config import config



# ============================================================
# Tokenizer initialization
# ============================================================

def get_tokenizer():

    """
    Initialize pretrained tokenizer

    """

    tokenizer = AutoTokenizer.from_pretrained(
        config.MODEL_NAME,
        use_fast=True
    )


    return tokenizer





# ============================================================
# Basic tokenize
# ============================================================

def tokenize_text(
        tokenizer,
        text
):

    """
    Tokenize one text

    Return:

    input_ids
    attention_mask
    offset_mapping

    """



    encoded = tokenizer(
        text,

        max_length=config.MAX_LENGTH,

        truncation=True,

        padding="max_length",

        return_offsets_mapping=True,

        return_tensors=None
    )


    return encoded





# ============================================================
# Long document sliding window
# ============================================================

def tokenize_with_chunks(
        tokenizer,
        text
):

    """
    Sliding window tokenizer


    Example:

    text length > 512


    output:

    [
      chunk1,
      chunk2,
      chunk3
    ]


    Each chunk contains:

    input_ids

    attention_mask

    offset_mapping


    """



    encoded = tokenizer(

        text,

        max_length=config.MAX_LENGTH,

        stride=config.STRIDE,

        truncation=True,

        padding="max_length",

        return_overflowing_tokens=True,

        return_offsets_mapping=True,

        return_attention_mask=True

    )



    chunks=[]



    for i in range(
        len(encoded["input_ids"])
    ):


        chunk={

            "input_ids":
                encoded[
                    "input_ids"
                ][i],


            "attention_mask":
                encoded[
                    "attention_mask"
                ][i],


            "offset_mapping":
                encoded[
                    "offset_mapping"
                ][i]

        }


        chunks.append(
            chunk
        )



    # limit maximum chunks

    if len(chunks) > config.MAX_CHUNKS:


        chunks = chunks[
            :config.MAX_CHUNKS
        ]



    return chunks





# ============================================================
# Offset cleaning
# ============================================================

def clean_offset_mapping(
        offsets
):

    """
    Convert tokenizer offset output


    Remove special tokens:

    CLS
    SEP
    PAD


    """



    cleaned=[]


    for start,end in offsets:


        if start==0 and end==0:

            cleaned.append(
                (-1,-1)
            )


        else:

            cleaned.append(
                (
                    start,
                    end
                )
            )


    return cleaned





# ============================================================
# Evidence alignment helper
# ============================================================

def align_evidence_to_chunk(
        chunk,
        evidence_spans
):

    """
    Convert character evidence spans
    into token labels for one chunk


    """



    offsets = chunk[
        "offset_mapping"
    ]


    seq_len=len(
        offsets
    )


    start_labels=np.zeros(
        seq_len,
        dtype=np.float32
    )


    end_labels=np.zeros(
        seq_len,
        dtype=np.float32
    )



    for char_start,char_end in evidence_spans:


        token_start=None

        token_end=None



        for idx,(s,e) in enumerate(
            offsets
        ):



            if s==0 and e==0:

                continue



            if (
                e > char_start
                and
                s < char_end
            ):


                if token_start is None:

                    token_start=idx


                token_end=idx



        if token_start is not None:


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
# Batch padding helper
# ============================================================

def pad_chunks(
        chunks
):

    """
    Ensure every sample has
    same number of chunks


    """



    while len(chunks)<config.MAX_CHUNKS:


        empty={

            "input_ids":
                [
                    0
                ] *
                config.MAX_LENGTH,


            "attention_mask":
                [
                    0
                ] *
                config.MAX_LENGTH,


            "offset_mapping":
                [
                    (0,0)
                ] *
                config.MAX_LENGTH

        }


        chunks.append(
            empty
        )



    return chunks[:config.MAX_CHUNKS]





# ============================================================
# Full preprocessing pipeline
# ============================================================

def encode_post(
        tokenizer,
        text
):

    """
    Complete encoding interface


    Used by cache_builder.py


    """



    chunks = tokenize_with_chunks(
        tokenizer,
        text
    )


    chunks = pad_chunks(
        chunks
    )


    return chunks





# ============================================================
# Debug
# ============================================================

if __name__ == "__main__":


    tokenizer=get_tokenizer()


    text = (
        "I want to kill myself "
        "and I do not want to wake up."
    )


    output=encode_post(
        tokenizer,
        text
    )


    print(
        len(output)
    )


    print(
        output[0].keys()
    )