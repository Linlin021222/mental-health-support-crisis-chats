"""Definition-grounded local-LLM augmentation for Task 2 tail factors.

Generation uses no real posts, so the same synthetic pool is safe in every
user-disjoint fold.  A generic zero-shot NLI model (not PFA-fine-tuned) ranks
examples by target-definition support and confusion-label separation before a
TF-IDF classifier distils the accepted examples.
"""
from __future__ import annotations

import json
import re

import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from configs.config import config
from inference.factor_nli import _rank_decode
from preprocess.factor_semantic_bank_v15 import FACTOR_CLASSIFIER_PROMPTS, FACTOR_SEMANTIC_BANK
from preprocess.preprocess import load_train_data
from trainer.factor_cross_encoder_v2 import CONFUSION_GROUPS
from trainer.factor_llm_lexical_v6 import (
    _clean_examples, _current_v3_probability, _fit_predict, _parse_array,
    _select_label_weights,
)
from trainer.factor_sentence_evidence_v16 import _score_pairs
from trainer.task1_local_counterfactual_v56 import _load_model
from utils.seed import seed_everything


OUTPUT = config.OUTPUT_DIR / "factor_tail_augmentation_v22"
RAW_FILE = OUTPUT / "raw_candidates.json"
SYNTHETIC_FILE = OUTPUT / "filtered_synthetic.json"
OOF_FILE = OUTPUT / "oof_predictions.npz"
RESULTS = OUTPUT / "cv_results.json"
TRAINING_VERSION = "definition-boundary-filtered-tail-augmentation-v22b"
SEED = 222222
TAIL_MAX_SUPPORT = 80
TARGET_PER_TAIL = 52
NEW_PER_TAIL = 36
GENERATION_BATCH = 12
WEIGHT_GRID = (0., .05, .10, .20, .30, .50)


def _parse_generated(raw):
    """Parse complete JSON or recover complete strings from a truncated array."""
    values = _parse_array(raw)
    if values:
        return values, "complete_json"
    recovered = []
    for match in re.finditer(r'"((?:\\.|[^"\\])*)"', raw):
        try:
            recovered.append(json.loads('"' + match.group(1) + '"'))
        except json.JSONDecodeError:
            continue
    if recovered:
        return recovered, "partial_json_recovery"
    # Last-resort support for a model returning numbered lines.
    for line in raw.splitlines():
        value = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip().strip('"')
        if 7 <= len(value.split()) <= 90:
            recovered.append(value)
    return recovered, "line_recovery" if recovered else "failed"


def _confusions(label):
    result = set()
    for group in CONFUSION_GROUPS:
        if label in group:
            result.update(x for x in group if x != label)
    return sorted(result)


@torch.inference_mode()
def generate(force=False):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if SYNTHETIC_FILE.exists() and not force:
        payload = json.loads(SYNTHETIC_FILE.read_text(encoding="utf-8"))
        if payload.get("training_version") == TRAINING_VERSION:
            print(f"V22 resumed {sum(len(x) for x in payload['examples'])} filtered examples")
            return payload
    frame = load_train_data().reset_index(drop=True)
    targets = np.vstack(frame.factor_vector.to_numpy()).astype(np.int8)
    supports = targets.sum(0).astype(int)
    tail_labels = np.flatnonzero(supports < TAIL_MAX_SUPPORT).tolist()
    old_path = config.OUTPUT_DIR / "factor_llm_lexical_v6" / "synthetic.json"
    old = json.loads(old_path.read_text(encoding="utf-8"))["examples"]

    if RAW_FILE.exists() and not force:
        raw_payload = json.loads(RAW_FILE.read_text(encoding="utf-8"))
        if raw_payload.get("training_version") == TRAINING_VERSION:
            candidates = raw_payload["examples"]
            audits = raw_payload.get("generation_audit", [])
            completed_labels = set(raw_payload.get("completed_labels", []))
            print(f"V22 resumed raw generation: completed={len(completed_labels)}/"
                  f"{len(tail_labels)}", flush=True)
        else:
            candidates = [list(values) if i in tail_labels else []
                          for i, values in enumerate(old)]
            audits = []; completed_labels = set()
    else:
        candidates = [list(values) if i in tail_labels else []
                      for i, values in enumerate(old)]
        audits = []; completed_labels = set()

    pending_labels = [label for label in tail_labels if label not in completed_labels]
    if pending_labels:
        if not torch.cuda.is_available():
            raise RuntimeError("V22 local Qwen generation requires CUDA")
        seed_everything(SEED)
        model, tokenizer = _load_model()
        for label in pending_labels:
            entry = FACTOR_SEMANTIC_BANK[label]
            confusing = _confusions(label)
            exclusion = "\n".join(
                f"- NOT {config.FACTOR_LABELS[x]}: {FACTOR_SEMANTIC_BANK[x]['formal']}"
                for x in confusing
            ) or "- Avoid adding unrelated factors."
            seen = {" ".join(x.split()).casefold() for x in candidates[label]}
            accepted = []
            attempts = int(np.ceil(NEW_PER_TAIL / GENERATION_BATCH))
            for attempt in range(attempts):
                requested = min(GENERATION_BATCH, NEW_PER_TAIL - len(accepted))
                if requested <= 0:
                    break
                prompt = (
                    "Create a JSON array of natural, first-person Reddit-style research snippets "
                    "for a multi-label mental-health classifier. Every snippet must clearly imply "
                    "the requested target factor, but must not contain the taxonomy label itself. "
                    "Use a balanced mixture of direct and subtle/implicit expression, varied ages, "
                    "situations, spelling and tone. Express only the target factor; do not introduce "
                    "the excluded/confusing factors below. Keep suicide content non-graphic and never "
                    "include method instructions, quantities or URLs. Each snippet must be 12-80 words. "
                    f"Return exactly {requested} strings and no commentary.\n\n"
                    f"TARGET: {config.FACTOR_LABELS[label]}\n"
                    f"Formal meaning: {entry['formal']}\n"
                    f"Direct expressions: {entry['direct']}\n"
                    f"Implicit expressions: {entry['implicit']}\n"
                    f"Boundary: {entry['distinction']}\n"
                    f"Exclude: {entry['negative']}\n"
                    f"Confusing labels to avoid:\n{exclusion}"
                )
                rendered = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}], tokenize=False,
                    add_generation_prompt=True,
                )
                encoded = tokenizer(
                    rendered, return_tensors="pt", truncation=True, max_length=900,
                ).to("cuda")
                torch.manual_seed(SEED + 100 * label + attempt)
                generated = model.generate(
                    **encoded, max_new_tokens=1500, do_sample=True,
                    temperature=.82, top_p=.92, repetition_penalty=1.08,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                raw = tokenizer.decode(
                    generated[0, encoded.input_ids.shape[1]:], skip_special_tokens=True,
                )
                values, parse_mode = _parse_generated(raw)
                cleaned = _clean_examples(
                    values, config.FACTOR_LABELS[label], seen,
                )
                accepted.extend(cleaned)
                audits.append({
                    "label": config.FACTOR_LABELS[label], "attempt": attempt,
                    "requested": requested, "parsed": len(values),
                    "format_accepted": len(cleaned), "parse_mode": parse_mode,
                    "output_characters": len(raw),
                })
            candidates[label].extend(accepted[:NEW_PER_TAIL])
            completed_labels.add(label)
            print(f"V22 generated {config.FACTOR_LABELS[label]}: "
                  f"old={len(old[label])}, new={len(accepted[:NEW_PER_TAIL])}", flush=True)
            # Crash-safe checkpoint: a stopped generation resumes at the next
            # label instead of discarding several minutes of local-LLM work.
            RAW_FILE.write_text(json.dumps({
                "training_version": TRAINING_VERSION,
                "tail_labels": tail_labels, "supports": supports.tolist(),
                "completed_labels": sorted(completed_labels),
                "examples": candidates, "generation_audit": audits,
            }, indent=2), encoding="utf-8")
        del model, tokenizer
        torch.cuda.empty_cache()

    # Generic NLI filtering is independent of every competition fold.
    device = torch.device(config.DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(
        config.FACTOR_NLI_MODEL_NAME, use_fast=True, local_files_only=True,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        config.FACTOR_NLI_MODEL_NAME, dtype=torch.float32, local_files_only=True,
    ).to(device)
    filtered = [[] for _ in range(config.NUM_FACTORS)]
    filter_audit = []
    for label in tail_labels:
        texts = candidates[label]
        confusing = _confusions(label)
        target_scores = _score_pairs(
            model, tokenizer, texts,
            [FACTOR_CLASSIFIER_PROMPTS[label][4]] * len(texts), device,
            f"V22 target {label}",
        )
        if confusing:
            premises, hypotheses, mapping = [], [], []
            for row, text in enumerate(texts):
                for other in confusing:
                    premises.append(text)
                    hypotheses.append(FACTOR_CLASSIFIER_PROMPTS[other][4])
                    mapping.append(row)
            other_scores = _score_pairs(
                model, tokenizer, premises, hypotheses, device,
                f"V22 exclusions {label}",
            )
            maximum_other = np.zeros(len(texts), np.float32)
            for row, score in zip(mapping, other_scores):
                maximum_other[row] = max(maximum_other[row], float(score))
        else:
            maximum_other = np.zeros(len(texts), np.float32)
        quality = target_scores - .60 * maximum_other
        order = np.argsort(-quality, kind="stable")
        chosen = order[:min(TARGET_PER_TAIL, len(order))]
        filtered[label] = [texts[int(i)] for i in chosen]
        filter_audit.append({
            "label": config.FACTOR_LABELS[label], "support": int(supports[label]),
            "candidates": len(texts), "kept": len(chosen),
            "mean_target_score_kept": float(target_scores[chosen].mean()),
            "mean_confusion_score_kept": float(maximum_other[chosen].mean()),
            "minimum_quality_kept": float(quality[chosen].min()),
        })
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    payload = {
        "training_version": TRAINING_VERSION,
        "uses_real_posts_as_generation_seeds": False,
        "filter_model_finetuned_on_competition": False,
        "tail_support_cutoff": TAIL_MAX_SUPPORT,
        "target_per_tail": TARGET_PER_TAIL,
        "examples": filtered,
        "filter_audit": filter_audit,
    }
    SYNTHETIC_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(filter_audit, indent=2), flush=True)
    return payload


def _synthetic_rows(payload):
    texts, labels = [], []
    for label, examples in enumerate(payload["examples"]):
        texts.extend(examples); labels.extend([label] * len(examples))
    return texts, np.asarray(labels, dtype=np.int64)


def cross_validate():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    payload = generate()
    synthetic_texts, synthetic_labels = _synthetic_rows(payload)
    frame = load_train_data().reset_index(drop=True)
    targets = np.vstack(frame.factor_vector.to_numpy()).astype(np.int8)
    folds = list(StratifiedGroupKFold(
        n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED,
    ).split(np.zeros(len(frame)), frame.risk_label, frame.anon_user_id))
    lexical = np.zeros_like(targets, dtype=np.float32)
    for fold, (fit_idx, valid_idx) in enumerate(folds):
        print(f"V22 lexical fold {fold}", flush=True)
        lexical[valid_idx] = _fit_predict(
            frame.text.iloc[fit_idx].astype(str).tolist(), targets[fit_idx],
            frame.text.iloc[valid_idx].astype(str).tolist(),
            synthetic_texts, synthetic_labels,
        )
    base, _ = _current_v3_probability()
    baseline_prediction = np.zeros_like(targets, dtype=bool)
    candidate_prediction = np.zeros_like(targets, dtype=bool)
    fold_rows, weights = [], []
    for fold, (fit_idx, valid_idx) in enumerate(folds):
        prevalence = targets[fit_idx].mean(0)
        # Reuse V6's leak-free per-label choice, with the same fixed grid.
        selected = _select_label_weights(
            base, lexical, targets, fit_idx, prevalence,
        )
        probability = (1. - selected) * base[valid_idx] + selected * lexical[valid_idx]
        baseline_prediction[valid_idx] = _rank_decode(
            base[valid_idx], prevalence, 1.10,
        )
        candidate_prediction[valid_idx] = _rank_decode(
            probability, prevalence, 1.10,
        )
        weights.append(selected)
        fold_rows.append({
            "fold": fold,
            "baseline_macro_f1": float(f1_score(
                targets[valid_idx], baseline_prediction[valid_idx],
                average="macro", zero_division=0)),
            "candidate_macro_f1": float(f1_score(
                targets[valid_idx], candidate_prediction[valid_idx],
                average="macro", zero_division=0)),
        })
    baseline = float(f1_score(
        targets, baseline_prediction, average="macro", zero_division=0,
    ))
    candidate = float(f1_score(
        targets, candidate_prediction, average="macro", zero_division=0,
    ))
    result = {
        "training_version": TRAINING_VERSION,
        "synthetic_count": len(synthetic_texts), "folds": fold_rows,
        "baseline_crossfit_macro_f1": baseline,
        "candidate_crossfit_macro_f1": candidate,
        "delta": candidate - baseline,
        "median_label_weights": np.median(np.vstack(weights), axis=0).tolist(),
        "promising_for_graph_stack": bool(candidate >= baseline + .002),
        "adopted": False,
    }
    np.savez_compressed(OOF_FILE, probabilities=lexical, targets=targets)
    RESULTS.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == "__main__":
    cross_validate()
