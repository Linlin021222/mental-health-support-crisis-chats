"""Dual-prompt, verbatim-grounded local-Qwen factor rationale mining."""
from __future__ import annotations

import json
import re

import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold
from tqdm import tqdm

from configs.config import config
from preprocess.factor_semantic_bank_v15 import FACTOR_SEMANTIC_BANK
from preprocess.preprocess import load_train_data
from trainer.factor_llm_lexical_v6 import _current_v3_probability
from trainer.factor_sentence_evidence_v16 import _sentences
from trainer.task1_local_counterfactual_v56 import _load_model
from utils.seed import seed_everything


OUTPUT = config.OUTPUT_DIR / "factor_dual_rationale_v31"
PARTIAL_A = OUTPUT / "partial_a.json"
PARTIAL_B = OUTPUT / "partial_b.json"
RATIONALES = OUTPUT / "grounded_rationales.json"
SUMMARY = OUTPUT / "summary.json"
TRAINING_VERSION = "qwen7b-dual-grounded-factor-rationale-v31"
BATCH_SIZE = 3
EXAMPLES_PER_LABEL = 8
TARGET_LABELS = (1, 8, 11, 13, 14, 15, 16, 18, 19, 21, 22, 23)


def _normalise(text):
    return re.sub(r"\s+", " ", str(text)).strip()


def _select_cases(frame, targets, train_idx, probability):
    cases = []
    for label in TARGET_LABELS:
        positives = train_idx[targets[train_idx, label] == 1]
        # Mix prototypical and difficult positives. OOF scores are leak-free:
        # each training row was scored by a fold that held out its user.
        order = positives[np.argsort(probability[positives, label])]
        half = EXAMPLES_PER_LABEL // 2
        selected = np.concatenate([order[:half], order[-half:]])
        selected = np.unique(selected)
        if len(selected) < EXAMPLES_PER_LABEL:
            remainder = [x for x in order if x not in set(selected)]
            selected = np.concatenate([
                selected, np.asarray(remainder[:EXAMPLES_PER_LABEL-len(selected)])
            ])
        for row in selected[:EXAMPLES_PER_LABEL]:
            cases.append({
                "case_id": len(cases) + 1,
                "row_index": int(row),
                "row_id": str(frame.row_id.iloc[row]),
                "label_id": int(label),
                "factor": config.ID2FACTOR[label],
                "post": _normalise(frame.text.iloc[row]),
                "oof_probability": float(probability[row, label]),
            })
    return cases


def _case_text(case):
    entry = FACTOR_SEMANTIC_BANK[case["label_id"]]
    return (
        f"CASE {case['case_id']}\n"
        f"FACTOR: {case['factor']}\n"
        f"DEFINITION: {entry['formal']}\n"
        f"DIRECT OR IMPLICIT EXPRESSION: {entry['direct']} {entry['implicit']}\n"
        f"BOUNDARY: {entry['distinction']}\n"
        f"NOT SUFFICIENT: {entry['negative']}\n"
        f"POST: {case['post'][:2600]}"
    )


def _prompt(batch, perspective):
    cases = "\n\n".join(_case_text(case) for case in batch)
    if perspective == "a":
        instruction = (
            "Act as a conservative taxonomy annotator. Decide whether the author's post "
            "contains the factor under the exact definition and boundary."
        )
    else:
        instruction = (
            "Act as an independent evidence auditor. First try to disprove the factor using "
            "the NOT SUFFICIENT and BOUNDARY rules; accept it only if explicit post wording survives."
        )
    return (
        f"{instruction} This is research annotation, not clinical advice. For each case return "
        "exactly one JSON object in a JSON array with keys: id (integer), present (true/false), "
        "confidence (0 to 1), evidence (one shortest verbatim quote copied from POST, or empty "
        "when absent), and reason (maximum 18 words). Do not paraphrase evidence. Do not use "
        "general suicide language as proof of an unrelated factor. No Markdown.\n\n" + cases
    )


def _parse(raw, expected):
    text = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    left, right = text.find("["), text.rfind("]")
    if left < 0 or right <= left:
        return None
    try:
        rows = json.loads(text[left:right + 1])
    except json.JSONDecodeError:
        return None
    result = {}
    for row in rows if isinstance(rows, list) else []:
        try:
            identifier = int(row["id"])
            present = bool(row["present"])
            confidence = float(row["confidence"])
        except (KeyError, TypeError, ValueError):
            continue
        result[identifier] = {
            "present": present,
            "confidence": float(np.clip(confidence, 0, 1)),
            "evidence": _normalise(row.get("evidence", "")),
            "reason": _normalise(row.get("reason", ""))[:200],
            "raw": raw,
        }
    return result if all(identifier in result for identifier in expected) else None


@torch.inference_mode()
def _judge(model, tokenizer, cases, perspective, path, force=False):
    partial = {} if force or not path.exists() else json.loads(path.read_text(encoding="utf-8"))
    pending = [case for case in cases if str(case["case_id"]) not in partial]
    for start in tqdm(range(0, len(pending), BATCH_SIZE), desc=f"V31 Qwen judge {perspective.upper()}"):
        batch = pending[start:start+BATCH_SIZE]
        prompt = _prompt(batch, perspective)
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False,
            add_generation_prompt=True,
        )
        encoded = tokenizer(
            rendered, return_tensors="pt", truncation=True, max_length=7168
        ).to("cuda")
        generated = model.generate(
            **encoded, max_new_tokens=500, do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        raw = tokenizer.decode(
            generated[0, encoded.input_ids.shape[1]:], skip_special_tokens=True
        )
        expected = [case["case_id"] for case in batch]
        parsed = _parse(raw, expected)
        if parsed is None:
            parsed = {}
            for case in batch:
                single_prompt = _prompt([case], perspective)
                single_rendered = tokenizer.apply_chat_template(
                    [{"role": "user", "content": single_prompt}], tokenize=False,
                    add_generation_prompt=True,
                )
                single = tokenizer(
                    single_rendered, return_tensors="pt", truncation=True, max_length=4096
                ).to("cuda")
                output = model.generate(
                    **single, max_new_tokens=180, do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                single_raw = tokenizer.decode(
                    output[0, single.input_ids.shape[1]:], skip_special_tokens=True
                )
                item = _parse(single_raw, [case["case_id"]])
                parsed[case["case_id"]] = (item or {case["case_id"]: {
                    "present": False, "confidence": 0., "evidence": "",
                    "reason": "format failure", "raw": single_raw,
                }})[case["case_id"]]
        for identifier, value in parsed.items():
            partial[str(identifier)] = value
        path.write_text(json.dumps(partial, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"V31 judge {perspective}: {len(partial)}/{len(cases)}", flush=True)
    return partial


def _locate(post, evidence):
    post_normal = _normalise(post)
    evidence_normal = _normalise(evidence)
    if not evidence_normal:
        return None
    position = post_normal.casefold().find(evidence_normal.casefold())
    if position < 0:
        return None
    return post_normal[position:position + len(evidence_normal)]


def _compatible(first, second):
    a, b = first.casefold(), second.casefold()
    if a in b or b in a:
        return True
    a_tokens, b_tokens = set(a.split()), set(b.split())
    return len(a_tokens & b_tokens) / max(1, min(len(a_tokens), len(b_tokens))) >= .55


def generate(force=False):
    if not torch.cuda.is_available():
        raise RuntimeError("Factor dual rationale V31 requires CUDA")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    seed_everything(config.SEED + 3100)
    frame = load_train_data().reset_index(drop=True)
    targets = np.vstack(frame.factor_vector.to_numpy()).astype(np.int8)
    risk = frame.risk_label.to_numpy(dtype=np.int64)
    groups = frame.anon_user_id.astype(str).to_numpy()
    train_idx, valid_idx = next(StratifiedGroupKFold(
        n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED,
    ).split(np.zeros(len(frame)), risk, groups))
    probability, _ = _current_v3_probability()
    cases = _select_cases(frame, targets, train_idx, probability)
    model, tokenizer = _load_model()
    first = _judge(model, tokenizer, cases, "a", PARTIAL_A, force=force)
    second = _judge(model, tokenizer, cases, "b", PARTIAL_B, force=force)
    del model
    torch.cuda.empty_cache()

    accepted, audit = [], []
    reasons = {}
    for case in cases:
        key = str(case["case_id"]); a, b = first[key], second[key]
        evidence_a = _locate(case["post"], a["evidence"])
        evidence_b = _locate(case["post"], b["evidence"])
        reason = "accepted"
        if not a["present"] or not b["present"]:
            reason = "judge_absent"
        elif min(a["confidence"], b["confidence"]) < .80:
            reason = "low_confidence"
        elif evidence_a is None or evidence_b is None:
            reason = "non_verbatim"
        elif not _compatible(evidence_a, evidence_b):
            reason = "evidence_disagreement"
        elif max(len(evidence_a.split()), len(evidence_b.split())) > 80:
            reason = "evidence_too_long"
        record = {
            **{k: v for k, v in case.items() if k != "post"},
            "post": case["post"], "judge_a": a, "judge_b": b,
            "verbatim_a": evidence_a, "verbatim_b": evidence_b,
            "filter_decision": reason,
        }
        audit.append(record); reasons[reason] = reasons.get(reason, 0) + 1
        if reason == "accepted":
            # Prefer the shorter faithful span for sentence-level training.
            record["evidence"] = min((evidence_a, evidence_b), key=lambda x: len(x.split()))
            accepted.append(record)
    payload = {
        "training_version": TRAINING_VERSION,
        "scope": "strict fold0 training users only",
        "target_labels": [config.ID2FACTOR[x] for x in TARGET_LABELS],
        "cases": len(cases), "accepted": len(accepted),
        "filter_counts": reasons, "records": accepted,
        "audit": audit,
    }
    RATIONALES.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {key: value for key, value in payload.items() if key not in {"records", "audit"}}
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return payload


if __name__ == "__main__":
    generate()
