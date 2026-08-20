"""Local Qwen2.5-7B counterfactual augmentation for Task 1 (V56).

Generation is deliberately non-graphic and restricted to training users.  A
strict validator requires a verbatim evidence span and definition-consistent
boundary cues before any generated row may be used for model training.
"""
from __future__ import annotations

import json
import re
from difflib import SequenceMatcher

import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from configs.config import config
from baseline import ATTEMPT_CUE, BEHAVIOR_CUE, IDEATION_CUE
from inference.task1_evidence_v4 import (
    EXTENDED_ATTEMPT_CUE, EXTENDED_BEHAVIOR_CUE, EXTENDED_IDEATION_CUE,
)
from preprocess.preprocess import load_train_data
from utils.seed import seed_everything


OUTPUT = config.OUTPUT_DIR / "task1_local_counterfactual_v56"
SYNTHETIC_FILE = OUTPUT / "synthetic.json"
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
TRANSFORMS = {
    "ideation_to_behavior": {
        "source": 1, "target": 2,
        "instruction": (
            "Rewrite it so the author still has suicidal thoughts and now explicitly "
            "states a near-future plan, but do not name a method, object, place, dose, "
            "or operational step. A generic phrase such as 'I plan to end my life soon' is enough."
        ),
    },
    "behavior_to_ideation": {
        "source": 2, "target": 1,
        "instruction": (
            "Rewrite it so the author explicitly wants to die, but remove every plan, "
            "method, means, access, preparation, timeframe, self-harm action, and past attempt."
        ),
    },
    "behavior_to_attempt": {
        "source": 2, "target": 3,
        "instruction": (
            "Rewrite it so the author explicitly says a recent suicide attempt already "
            "happened and they survived. Keep it non-graphic and do not name a method, dose, or steps."
        ),
    },
}
RISK_NAME = {1: "Ideation", 2: "Behavior", 3: "Attempt"}
ACTIONABLE = re.compile(
    r"(?i)(\b\d+(?:\.\d+)?\s*(?:mg|g|grams?|pills?|tablets?|ml)\b|"
    r"\b(?:overdose|hanging|hang myself|shoot myself|gun|rope|razor|knife|bridge|"
    r"sodium nitrite|bleed out|cliff|highway|train tracks?)\b|https?://)"
)
ANCHORS = {
    1: (
        "I want to die",
        "I wish I were dead",
        "I don't want to live anymore",
        "I keep thinking that I want to die",
        "I wish I was dead",
        "I feel that I should end my life",
    ),
    2: (
        "I plan to end my life soon",
        "I am going to end my life soon",
        "I planned to kill myself soon",
        "I plan to kill myself soon",
        "I am going to kill myself soon",
        "I planned to end my life soon",
    ),
    3: (
        "I tried to end my life recently",
        "I survived a suicide attempt recently",
        "After my suicide attempt, I am still here",
        "I attempted suicide recently",
        "I survived my suicide attempt",
        "I tried to kill myself recently",
    ),
}


def _load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    quantization = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=quantization, device_map={"": 0},
        local_files_only=True,
    )
    model.eval()
    return model, tokenizer


def _prompt(source_text, source_evidence, transform, anchor):
    spec = TRANSFORMS[transform]
    target = RISK_NAME[spec["target"]]
    context = " ".join(str(source_text).split())
    # The old evidence expresses the source label and strongly encourages
    # copying. Remove it before asking for the counterfactual boundary.
    for phrase in source_evidence:
        context = re.sub(re.escape(str(phrase)), " ", context, flags=re.IGNORECASE)
    context = ACTIONABLE.sub(" ", context)
    context = " ".join(context.split())[:650]
    return (
        "Create one non-graphic, first-person Reddit-style research example. "
        f"Target risk label: {target}. {spec['instruction']} "
        "Preserve ordinary emotional and life context, but change the suicide-risk boundary as requested. "
        f"The rewritten post MUST contain this exact sentence once: {anchor}. "
        f"The JSON evidence value MUST be exactly: {anchor}. "
        "Return exactly one JSON object with string keys post and evidence. Do not use a list, semicolons, "
        "Markdown, or commentary.\n"
        f"Source context with the old risk evidence removed: {context}\n"
        f"Required output shape: {{\"post\": \"rewritten context ... {anchor}.\", "
        f"\"evidence\": \"{anchor}\"}}"
    )


def _json_object(raw):
    text = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no_json_object")
    value = json.loads(text[start:end + 1])
    if not isinstance(value, dict):
        raise ValueError("json_not_object")
    return value


def _validate(value, source_text, target, anchor):
    post = " ".join(str(value.get("post", "")).split()).strip()
    evidence = " ".join(str(value.get("evidence", "")).split()).strip().strip("; ")
    if not 12 <= len(post.split()) <= 190:
        return None, "post_length"
    if not 2 <= len(evidence.split()) <= 18:
        return None, "evidence_length"
    if evidence.casefold().rstrip(".!") != anchor.casefold().rstrip(".!"):
        return None, "wrong_anchor"
    position = post.casefold().find(evidence.casefold())
    if position < 0:
        return None, "evidence_not_verbatim"
    evidence = post[position:position + len(evidence)]
    if ACTIONABLE.search(post):
        return None, "actionable_detail"
    if SequenceMatcher(None, post.casefold(), str(source_text).casefold()).ratio() > .96:
        return None, "near_duplicate"
    attempt = bool(EXTENDED_ATTEMPT_CUE.search(post) or ATTEMPT_CUE.search(post))
    behavior = bool(EXTENDED_BEHAVIOR_CUE.search(post) or BEHAVIOR_CUE.search(post))
    ideation = bool(EXTENDED_IDEATION_CUE.search(post) or IDEATION_CUE.search(post))
    if target == 1 and (not ideation or behavior or attempt):
        return None, "ideation_boundary"
    if target == 2 and (not behavior or attempt):
        return None, "behavior_boundary"
    if target == 3 and not attempt:
        return None, "attempt_boundary"
    return {"post": post, "evidence": evidence}, None


@torch.inference_mode()
def generate_counterfactuals(target_per_transform=18, force=False):
    if not torch.cuda.is_available():
        raise RuntimeError("V56 local generation requires CUDA")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if SYNTHETIC_FILE.exists() and not force:
        saved = json.loads(SYNTHETIC_FILE.read_text(encoding="utf-8"))
        print(f"V56 generation resumed: {len(saved['accepted'])} accepted", flush=True)
        return saved
    seed_everything(config.SEED + 5656)
    frame = load_train_data().reset_index(drop=True)
    labels = frame.risk_label.to_numpy(dtype=np.int64)
    groups = frame.anon_user_id.astype(str).to_numpy()
    train_idx, _ = next(StratifiedGroupKFold(
        n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED,
    ).split(np.zeros(len(frame)), labels, groups))
    rng = np.random.default_rng(config.SEED + 5656)
    jobs = []
    for transform, spec in TRANSFORMS.items():
        candidates = np.asarray([
            int(index) for index in train_idx
            if labels[int(index)] == spec["source"]
            and bool(frame.iloc[int(index)].evidence)
            and 20 <= len(str(frame.iloc[int(index)].text).split()) <= 220
        ])
        rng.shuffle(candidates)
        for index in candidates[:target_per_transform]:
            jobs.append((transform, int(index)))
    model, tokenizer = _load_model()
    accepted, rejected = [], []
    torch.cuda.reset_peak_memory_stats()
    for job_number, (transform, index) in enumerate(tqdm(jobs, desc="V56 local generation")):
        row = frame.iloc[index]
        target = int(TRANSFORMS[transform]["target"])
        anchor = ANCHORS[target][job_number % len(ANCHORS[target])]
        prompt = _prompt(row.text, row.evidence, transform, anchor)
        messages = [{"role": "user", "content": prompt}]
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        encoded = tokenizer(
            rendered, return_tensors="pt", truncation=True, max_length=320,
        ).to("cuda")
        torch.manual_seed(config.SEED + 5656 + job_number)
        output = model.generate(
            **encoded, max_new_tokens=170, do_sample=True, temperature=.65,
            top_p=.90, repetition_penalty=1.05,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
        )
        raw = tokenizer.decode(output[0, encoded.input_ids.shape[1]:], skip_special_tokens=True)
        try:
            parsed = _json_object(raw)
            clean, reason = _validate(parsed, row.text, target, anchor)
        except Exception as error:
            clean, reason = None, f"parse:{type(error).__name__}:{error}"
        audit = {"transform": transform, "source_index": index,
                 "source_row_id": str(row.row_id),
                 "source_risk": int(labels[index]),
                 "target_risk": int(TRANSFORMS[transform]["target"])}
        if clean is None:
            rejected.append({**audit, "reason": reason, "raw": raw})
        else:
            factors = np.asarray(row.factor_vector, dtype=np.float32).copy()
            factors[9] = 1.0
            factors[17] = 1.0 if TRANSFORMS[transform]["target"] == 2 else 0.0
            accepted.append({**audit, **clean,
                             "factor_vector": factors.astype(float).tolist()})
        print(f"V56 {job_number + 1}/{len(jobs)} accepted={len(accepted)} "
              f"rejected={len(rejected)}", flush=True)
    payload = {
        "training_version": "task1-local-qwen7b-counterfactual-v56",
        "model": MODEL_NAME, "scope": "outer training users only",
        "requested_per_transform": int(target_per_transform),
        "accepted": accepted, "rejected": rejected,
        "acceptance_rate": float(len(accepted) / max(1, len(jobs))),
        "peak_memory_gb": float(torch.cuda.max_memory_allocated() / 1024 ** 3),
    }
    SYNTHETIC_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items()
                      if key not in ("accepted", "rejected")}, indent=2), flush=True)
    del model
    torch.cuda.empty_cache()
    return payload


if __name__ == "__main__":
    generate_counterfactuals()
