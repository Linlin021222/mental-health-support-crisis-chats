"""Deterministic local Qwen7B factor judge on a strict user holdout (V7)."""
from __future__ import annotations

import json

import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from tqdm import tqdm

from configs.config import config
from inference.factor_nli import _rank_decode
from preprocess.factor_prototypes import FACTOR_PROTOTYPES
from preprocess.preprocess import load_train_data
from trainer.factor_llm_lexical_v6 import _current_v3_probability
from trainer.task1_local_counterfactual_v56 import _load_model
from utils.seed import seed_everything


OUTPUT = config.OUTPUT_DIR / "factor_qwen_direct_v7"
PARTIAL_FILE = OUTPUT / "strict_partial.json"
PREDICTION_FILE = OUTPUT / "strict_predictions.npz"
RESULTS_FILE = OUTPUT / "strict_results.json"
TRAINING_VERSION = "factor-qwen7b-direct-judge-v7"
SEED = 707070
BATCH_POSTS = 3
FIXED_WEIGHT = .10


def _taxonomy():
    lines = []
    for index, (label, prototypes) in enumerate(zip(config.FACTOR_LABELS, FACTOR_PROTOTYPES)):
        lines.append(f"{index}: {label} — {prototypes[0]}")
    return "\n".join(lines)


def _shorten(text, maximum=1400):
    text = " ".join(str(text).split())
    if len(text) <= maximum:
        return text
    half = maximum // 2
    return text[:half] + " ... [middle omitted] ... " + text[-half:]


def _parse(raw, expected):
    text = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    left, right = text.find("["), text.rfind("]")
    if left < 0 or right <= left:
        return None
    try:
        value = json.loads(text[left:right + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(value, list):
        return None
    result = {}
    for row in value:
        if not isinstance(row, dict) or "id" not in row or "labels" not in row:
            continue
        try:
            identifier = int(row["id"])
            labels = sorted({int(item) for item in row["labels"]
                             if 0 <= int(item) < config.NUM_FACTORS})
        except (TypeError, ValueError):
            continue
        result[identifier] = labels
    return result if all(identifier in result for identifier in expected) else None


@torch.inference_mode()
def predict_strict(force=False):
    if not torch.cuda.is_available():
        raise RuntimeError("Factor V7 requires CUDA")
    OUTPUT.mkdir(parents=True, exist_ok=True); seed_everything(SEED)
    frame = load_train_data().reset_index(drop=True)
    labels = frame.risk_label.to_numpy(dtype=np.int64)
    groups = frame.anon_user_id.astype(str).to_numpy()
    _, valid_idx = next(StratifiedGroupKFold(
        n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED,
    ).split(np.zeros(len(frame)), labels, groups))
    partial = {} if force or not PARTIAL_FILE.exists() else json.loads(
        PARTIAL_FILE.read_text(encoding="utf-8")
    )
    pending = [int(index) for index in valid_idx if str(int(index)) not in partial]
    if pending:
        model, tokenizer = _load_model(); taxonomy = _taxonomy()
        for start in tqdm(range(0, len(pending), BATCH_POSTS), desc="Factor V7 Qwen strict"):
            indices = pending[start:start + BATCH_POSTS]
            posts = "\n".join(
                f"POST {index}: {_shorten(frame.iloc[index].text)}" for index in indices
            )
            prompt = (
                "You are annotating Reddit posts for a research benchmark. Identify every suicide-related "
                "risk or protective factor explicitly supported by each post. Do not infer a factor merely "
                "because it is common in suicidal people. Distinguish absent support (10) from received "
                "support (19), hopelessness (3) from positive resilience (21), responsibility (22) from "
                "life meaning (23), stressful events (14) from trauma (15), and the author's own history "
                "(9) from another person's suicide (13). Return exactly a JSON array; each object must be "
                "{\"id\": integer, \"labels\": [integer IDs]}. Use [] when none. No explanations.\n\n"
                f"TAXONOMY:\n{taxonomy}\n\nPOSTS:\n{posts}"
            )
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False,
                add_generation_prompt=True,
            )
            encoded = tokenizer(rendered, return_tensors="pt", truncation=True,
                                max_length=4096).to("cuda")
            output = model.generate(
                **encoded, max_new_tokens=180, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            raw = tokenizer.decode(
                output[0, encoded.input_ids.shape[1]:], skip_special_tokens=True,
            )
            parsed = _parse(raw, indices)
            if parsed is None:
                # Preserve progress and make a conservative empty prediction;
                # the audit keeps the raw refusal/format failure visible.
                parsed = {index: [] for index in indices}
            for index in indices:
                partial[str(index)] = {"labels": parsed[index], "raw": raw}
            PARTIAL_FILE.write_text(json.dumps(partial, indent=2), encoding="utf-8")
            print(f"Factor V7 completed={len(partial)}/{len(valid_idx)}", flush=True)
        del model; torch.cuda.empty_cache()
    prediction = np.zeros((len(valid_idx), config.NUM_FACTORS), dtype=np.float32)
    failures = 0
    for position, index in enumerate(valid_idx):
        row = partial[str(int(index))]
        prediction[position, row["labels"]] = 1.; failures += int(not row["labels"])
    np.savez_compressed(PREDICTION_FILE, probabilities=prediction,
                        valid_indices=valid_idx)
    print(f"Factor V7 strict predictions ready: rows={len(valid_idx)}, empty={failures}")
    return prediction, valid_idx


def evaluate():
    if not PREDICTION_FILE.exists():
        predict_strict()
    saved = np.load(PREDICTION_FILE)
    qwen = saved["probabilities"].astype(np.float32)
    valid_idx = saved["valid_indices"].astype(int)
    frame = load_train_data().reset_index(drop=True)
    targets = np.vstack(frame.factor_vector.to_numpy()).astype(np.int8)
    train_idx = np.setdiff1d(np.arange(len(frame)), valid_idx)
    prevalence = targets[train_idx].mean(0)
    base, _ = _current_v3_probability(); base = base[valid_idx]
    baseline_prediction = _rank_decode(base, prevalence, 1.10)
    # Binary LLM decisions are softened so they only reorder the accepted V3
    # scores; global prevalence and cardinality remain anchored to real train.
    qwen_score = .15 + .70 * qwen
    candidate_probability = (1. - FIXED_WEIGHT) * base + FIXED_WEIGHT * qwen_score
    candidate_prediction = _rank_decode(candidate_probability, prevalence, 1.10)
    truth = targets[valid_idx]
    baseline = float(f1_score(truth, baseline_prediction, average="macro", zero_division=0))
    standalone = float(f1_score(truth, qwen, average="macro", zero_division=0))
    candidate = float(f1_score(truth, candidate_prediction, average="macro", zero_division=0))
    groups = frame.anon_user_id.astype(str).to_numpy()[valid_idx]
    users = np.unique(groups); rng = np.random.default_rng(SEED); deltas = []
    for _ in range(3000):
        sample = rng.choice(users, size=len(users), replace=True)
        positions = np.concatenate([np.flatnonzero(groups == user) for user in sample])
        old = f1_score(truth[positions], baseline_prediction[positions], average="macro", zero_division=0)
        new = f1_score(truth[positions], candidate_prediction[positions], average="macro", zero_division=0)
        deltas.append(float(new - old))
    deltas = np.asarray(deltas)
    bootstrap = {
        "mean_delta": float(deltas.mean()), "p05_delta": float(np.quantile(deltas, .05)),
        "p95_delta": float(np.quantile(deltas, .95)),
        "positive_fraction": float((deltas > 0).mean()),
    }
    adopted = bool(candidate >= baseline + .005 and bootstrap["positive_fraction"] >= .8)
    per_label = [{
        "label": config.ID2FACTOR[label], "support": int(truth[:, label].sum()),
        "baseline_f1": float(f1_score(truth[:, label], baseline_prediction[:, label], zero_division=0)),
        "qwen_f1": float(f1_score(truth[:, label], qwen[:, label], zero_division=0)),
        "candidate_f1": float(f1_score(truth[:, label], candidate_prediction[:, label], zero_division=0)),
    } for label in range(config.NUM_FACTORS)]
    payload = {
        "training_version": TRAINING_VERSION, "strict_fold": 0,
        "baseline_macro_f1": baseline, "qwen_standalone_macro_f1": standalone,
        "candidate_macro_f1": candidate, "fixed_weight": FIXED_WEIGHT,
        "changed_cells": int(np.sum(candidate_prediction != baseline_prediction)),
        "per_label": per_label, "user_cluster_bootstrap": bootstrap,
        "adopted": adopted,
    }
    RESULTS_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    return payload


if __name__ == "__main__":
    predict_strict(); evaluate()
