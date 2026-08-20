"""Retrieval-augmented local-Qwen audit of Task 2 boundary decisions.

Unlike the weak V7 all-label prompt, each request concerns one factor only and
contains its formal boundary plus leak-free, similar positive/negative examples
from outer-fold training users.  Qwen only reorders eight cases around each
pre-registered ambiguous label's decision boundary; the accepted V3 model and
training prevalence continue to determine all other predictions.
"""
from __future__ import annotations

import json
import re

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from tqdm import tqdm

from configs.config import config
from inference.factor_nli import _rank_decode
from preprocess.factor_semantic_bank_v15 import FACTOR_SEMANTIC_BANK
from preprocess.preprocess import load_train_data
from trainer.factor_llm_lexical_v6 import _current_v3_probability
from trainer.factor_sentence_evidence_cv_v27 import _user_bootstrap
from trainer.task1_local_counterfactual_v56 import _load_model
from utils.seed import seed_everything


OUTPUT = config.OUTPUT_DIR / "factor_fewshot_boundary_v34"
PARTIAL = OUTPUT / "partial.json"
PREDICTIONS = OUTPUT / "fold0_judgements.npz"
RESULTS = OUTPUT / "fold0_results.json"
TRAINING_VERSION = "retrieval-fewshot-qwen7b-boundary-v34"
TARGET_LABELS = (8, 9, 11, 13, 14, 15, 16, 19, 21, 22, 23)
BOUNDARY_EACH_SIDE = 4
DEMONSTRATIONS_PER_CLASS = 2
BLEND_WEIGHT = 0.15
TOPK_RATIO = 1.10
BATCH_SIZE = 2


def _normalise(text):
    return re.sub(r"\s+", " ", str(text)).strip()


def _shorten(text, maximum=900):
    text = _normalise(text)
    if len(text) <= maximum:
        return text
    half = maximum // 2
    return text[:half] + " ... [middle omitted] ... " + text[-half:]


def _rank(values):
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float32)
    ranks[order] = np.linspace(0., 1., len(values), dtype=np.float32)
    return ranks


def _select_cases(frame, targets, train_idx, valid_idx, probability):
    train_text = frame.text.iloc[train_idx].astype(str).tolist()
    valid_text = frame.text.iloc[valid_idx].astype(str).tolist()
    vectorizer = TfidfVectorizer(
        lowercase=True, strip_accents="unicode", sublinear_tf=True,
        ngram_range=(1, 2), min_df=2, max_df=.995, max_features=80000,
    )
    train_matrix = vectorizer.fit_transform(train_text)
    valid_matrix = vectorizer.transform(valid_text)
    prevalence = targets[train_idx].mean(0)
    cases = []
    for label in TARGET_LABELS:
        count = max(1, int(round(len(valid_idx) * prevalence[label] * TOPK_RATIO)))
        order = np.argsort(probability[valid_idx, label])[::-1]
        selected_positions = order[
            max(0, count - BOUNDARY_EACH_SIDE):
            min(len(order), count + BOUNDARY_EACH_SIDE)
        ]
        positive_local = np.flatnonzero(targets[train_idx, label] == 1)
        negative_local = np.flatnonzero(targets[train_idx, label] == 0)
        # Prefer reliable demonstrations: high-scoring annotated positives and
        # low-scoring negatives, while retrieval keeps them textually relevant.
        pos_pool = positive_local[
            np.argsort(probability[train_idx[positive_local], label])[::-1]
            [:min(80, len(positive_local))]
        ]
        neg_pool = negative_local[
            np.argsort(probability[train_idx[negative_local], label])
            [:min(240, len(negative_local))]
        ]
        for valid_position in selected_positions:
            similarity = valid_matrix[valid_position] @ train_matrix.T
            similarity = similarity.toarray().ravel()
            pos_demo = pos_pool[np.argsort(similarity[pos_pool])[::-1][
                :DEMONSTRATIONS_PER_CLASS
            ]]
            neg_demo = neg_pool[np.argsort(similarity[neg_pool])[::-1][
                :DEMONSTRATIONS_PER_CLASS
            ]]
            cases.append({
                "id": len(cases) + 1,
                "label_id": int(label),
                "factor": config.ID2FACTOR[label],
                "valid_position": int(valid_position),
                "row_index": int(valid_idx[valid_position]),
                "post": _normalise(valid_text[valid_position]),
                "positive_examples": [_shorten(train_text[x], 650) for x in pos_demo],
                "negative_examples": [_shorten(train_text[x], 650) for x in neg_demo],
                "base_probability": float(probability[valid_idx[valid_position], label]),
            })
    return cases, prevalence


def _case_text(case):
    bank = FACTOR_SEMANTIC_BANK[case["label_id"]]
    positives = "\n".join(
        f"PRESENT EXAMPLE {i + 1}: {text}"
        for i, text in enumerate(case["positive_examples"])
    )
    negatives = "\n".join(
        f"ABSENT EXAMPLE {i + 1}: {text}"
        for i, text in enumerate(case["negative_examples"])
    )
    return (
        f"CASE {case['id']}\nFACTOR: {case['factor']}\n"
        f"FORMAL DEFINITION: {bank['formal']}\n"
        f"DIRECT/IMPLICIT: {bank['direct']} {bank['implicit']}\n"
        f"DISTINGUISH FROM: {bank['distinction']}\n"
        f"NOT SUFFICIENT: {bank['negative']}\n"
        f"{positives}\n{negatives}\n"
        f"POST TO CLASSIFY: {_shorten(case['post'], 1800)}"
    )


def _prompt(batch):
    return (
        "You are a conservative benchmark annotator. Each case asks about ONE factor. "
        "Examples illustrate this dataset but never override the formal definition. Decide "
        "whether the POST TO CLASSIFY supports that factor. General distress or suicidality "
        "alone is insufficient. Return exactly a JSON array with one object per case: "
        '{"id": integer, "present": true/false, "confidence": number 0 to 1, '
        '"evidence": "shortest verbatim quote from POST TO CLASSIFY or empty"}. '
        "No Markdown and no explanation.\n\n" +
        "\n\n".join(_case_text(case) for case in batch)
    )


def _parse(raw, expected):
    text = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    left, right = text.find("["), text.rfind("]")
    # Qwen sometimes returns a single JSON object for a one-case retry even
    # when an array was requested. Accept that harmless schema variation.
    if left < 0 or right <= left:
        left, right = text.find("{"), text.rfind("}")
        if left < 0 or right <= left:
            return None
        candidate = text[left:right + 1]
    else:
        candidate = text[left:right + 1]
    try:
        rows = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    if isinstance(rows, dict):
        rows = [rows]
    result = {}
    for row in rows if isinstance(rows, list) else []:
        try:
            identifier = int(row["id"])
            present = bool(row["present"])
            confidence = float(np.clip(float(row["confidence"]), 0., 1.))
        except (KeyError, TypeError, ValueError):
            continue
        result[identifier] = {
            "present": present, "confidence": confidence,
            "evidence": _normalise(row.get("evidence", "")), "raw": raw,
        }
    return result if all(x in result for x in expected) else None


def _verbatim(post, evidence):
    return bool(evidence) and _normalise(evidence).casefold() in _normalise(post).casefold()


@torch.inference_mode()
def _generate(cases, force=False):
    partial = {} if force or not PARTIAL.exists() else json.loads(
        PARTIAL.read_text(encoding="utf-8")
    )
    pending = [
        case for case in cases
        if str(case["id"]) not in partial
        or partial[str(case["id"])].get("format_failure")
    ]
    if not pending:
        return partial
    model, tokenizer = _load_model()
    # Initial batches amortise generation. Failed batches are repaired one case
    # at a time because the observed failure mode is returning only the first
    # object of a two-case request.
    batch_size = 1 if partial else BATCH_SIZE
    for start in tqdm(range(0, len(pending), batch_size), desc="V34 Qwen boundary audit"):
        batch = pending[start:start + batch_size]
        prompt = _prompt(batch)
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False,
            add_generation_prompt=True,
        )
        encoded = tokenizer(
            rendered, return_tensors="pt", truncation=True, max_length=7168
        ).to("cuda")
        output = model.generate(
            **encoded, max_new_tokens=260, do_sample=False,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
        )
        raw = tokenizer.decode(
            output[0, encoded.input_ids.shape[1]:], skip_special_tokens=True
        )
        ids = [case["id"] for case in batch]
        parsed = _parse(raw, ids)
        if parsed is None:
            parsed = {case["id"]: {
                "present": False, "confidence": 0., "evidence": "",
                "raw": raw, "format_failure": True,
            } for case in batch}
        for case in batch:
            value = parsed[case["id"]]
            value["verbatim"] = _verbatim(case["post"], value.get("evidence", ""))
            partial[str(case["id"])] = value
        PARTIAL.write_text(json.dumps(partial, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"V34 audited {len(partial)}/{len(cases)}", flush=True)
    del model
    torch.cuda.empty_cache()
    return partial


def run(force=False):
    if not torch.cuda.is_available():
        raise RuntimeError("Factor few-shot boundary V34 requires CUDA")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    seed_everything(config.SEED + 3400)
    frame = load_train_data().reset_index(drop=True)
    targets = np.vstack(frame.factor_vector.to_numpy()).astype(np.int8)
    risk = frame.risk_label.to_numpy(dtype=np.int64)
    groups = frame.anon_user_id.astype(str).to_numpy()
    train_idx, valid_idx = next(StratifiedGroupKFold(
        n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED,
    ).split(np.zeros(len(frame)), risk, groups))
    current, _ = _current_v3_probability()
    cases, prevalence = _select_cases(frame, targets, train_idx, valid_idx, current)
    print(
        f"V34 strict fold0 boundary cases={len(cases)}, labels={len(TARGET_LABELS)}",
        flush=True,
    )
    judgement = _generate(cases, force=force)
    base = current[valid_idx]
    mixed = np.column_stack([_rank(base[:, label]) for label in range(config.NUM_FACTORS)])
    accepted = 0; present = 0; failures = 0
    for case in cases:
        value = judgement[str(case["id"])]
        # A present decision requires grounded evidence. An absent decision can
        # be useful without evidence. Format failures make no change.
        if value.get("format_failure"):
            failures += 1; continue
        if value["present"] and not value.get("verbatim"):
            continue
        confidence = float(value["confidence"])
        qwen = confidence if value["present"] else 1. - confidence
        row, label = case["valid_position"], case["label_id"]
        mixed[row, label] = (1. - BLEND_WEIGHT) * mixed[row, label] + BLEND_WEIGHT * qwen
        accepted += 1; present += int(value["present"])
    baseline_prediction = _rank_decode(base, prevalence, TOPK_RATIO)
    candidate_prediction = _rank_decode(mixed, prevalence, TOPK_RATIO)
    truth = targets[valid_idx]
    baseline = float(f1_score(truth, baseline_prediction, average="macro", zero_division=0))
    candidate = float(f1_score(truth, candidate_prediction, average="macro", zero_division=0))
    bootstrap = _user_bootstrap(
        truth, baseline_prediction, candidate_prediction, groups[valid_idx],
        seed=343434, draws=3000,
    )
    per_label = [{
        "label": config.ID2FACTOR[label], "support": int(truth[:, label].sum()),
        "audited": sum(case["label_id"] == label for case in cases),
        "baseline_f1": float(f1_score(
            truth[:, label], baseline_prediction[:, label], zero_division=0
        )),
        "candidate_f1": float(f1_score(
            truth[:, label], candidate_prediction[:, label], zero_division=0
        )),
    } for label in range(config.NUM_FACTORS)]
    np.savez_compressed(
        PREDICTIONS, mixed=mixed, valid_indices=valid_idx,
        training_version=TRAINING_VERSION,
    )
    payload = {
        "training_version": TRAINING_VERSION,
        "evaluation": "untouched outer user fold0; demonstrations use training users only",
        "fixed_policy": {
            "target_labels": [config.ID2FACTOR[x] for x in TARGET_LABELS],
            "boundary_each_side": BOUNDARY_EACH_SIDE,
            "demonstrations_per_class": DEMONSTRATIONS_PER_CLASS,
            "blend_weight": BLEND_WEIGHT, "topk_ratio": TOPK_RATIO,
        },
        "cases": len(cases), "accepted_judgements": accepted,
        "present_judgements": present, "format_failures": failures,
        "changed_cells": int(np.sum(baseline_prediction != candidate_prediction)),
        "baseline_macro_f1": baseline, "candidate_macro_f1": candidate,
        "delta": candidate - baseline, "user_cluster_bootstrap": bootstrap,
        "per_label": per_label,
        "promising_for_full_oof": bool(
            candidate >= baseline + .005 and bootstrap["positive_fraction"] >= .70
        ),
        "adopted": False,
    }
    RESULTS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    return payload


if __name__ == "__main__":
    run()
