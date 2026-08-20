"""Qwen-generated lexical distillation for long-tail suicide factors (V6).

The local LLM creates definition-grounded, single-factor Reddit snippets.  A
small word/character model distils those examples together with real training
posts.  All blend weights are selected cross-fit by user-disjoint OOF data.
"""
from __future__ import annotations

import json
import re

import joblib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

from baseline import _vectorizer
from configs.config import config
from inference.factor_nli import _rank_decode
from preprocess.factor_prototypes import FACTOR_PROTOTYPES
from preprocess.preprocess import load_train_data
from trainer.task1_local_counterfactual_v56 import _load_model
from utils.seed import seed_everything


OUTPUT = config.OUTPUT_DIR / "factor_llm_lexical_v6"
SYNTHETIC_FILE = OUTPUT / "synthetic.json"
OOF_FILE = OUTPUT / "oof_predictions.npz"
RESULTS_FILE = OUTPUT / "cv_results.json"
CALIBRATION_FILE = OUTPUT / "calibration.json"
FULL_MODEL_FILE = OUTPUT / "full_model.joblib"
TRAINING_VERSION = "factor-qwen7b-lexical-distillation-v6"
SEED = 606060
PER_LABEL = 16
POSITIVE_SYNTHETIC_WEIGHT = .35
NEGATIVE_SYNTHETIC_WEIGHT = .08
WEIGHT_GRID = (0., .05, .10, .20, .30, .50)


def _parse_array(raw):
    text = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    left, right = text.find("["), text.rfind("]")
    if left < 0 or right <= left:
        return []
    try:
        value = json.loads(text[left:right + 1])
    except json.JSONDecodeError:
        return []
    return value if isinstance(value, list) else []


def _clean_examples(values, label, seen):
    accepted = []
    literal = re.compile(re.escape(label), flags=re.IGNORECASE)
    dangerous_detail = re.compile(
        r"(?i)\b\d+(?:\.\d+)?\s*(?:mg|g|grams?|pills?|tablets?|ml)\b|https?://"
    )
    for value in values:
        if not isinstance(value, str):
            continue
        text = " ".join(value.split()).strip(" -\t\r\n\"")
        normalized = text.casefold()
        if not 7 <= len(text.split()) <= 85:
            continue
        if literal.search(text) or dangerous_detail.search(text):
            continue
        if normalized in seen:
            continue
        seen.add(normalized); accepted.append(text)
    return accepted


@torch.inference_mode()
def generate(force=False):
    if not torch.cuda.is_available():
        raise RuntimeError("Factor V6 generation requires CUDA")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if SYNTHETIC_FILE.exists() and not force:
        payload = json.loads(SYNTHETIC_FILE.read_text(encoding="utf-8"))
        print(f"Factor V6 generation resumed: {sum(len(x) for x in payload['examples'])} examples")
        return payload
    seed_everything(SEED)
    model, tokenizer = _load_model(); examples = []; audits = []
    torch.cuda.reset_peak_memory_stats()
    for label_index, (label, prototypes) in enumerate(zip(config.FACTOR_LABELS, FACTOR_PROTOTYPES)):
        accepted = []; seen = set(); attempt_audit = []
        for attempt in range(2):
            remaining = PER_LABEL - len(accepted)
            if remaining <= 0:
                break
            definitions = "\n".join(f"- {item}" for item in prototypes)
            prompt = (
                "Create natural first-person Reddit-style snippets for supervised mental-health "
                "factor classification. Each snippet must demonstrate ONLY the requested factor, "
                "using a concrete everyday situation. Do not mention the category name, taxonomy, "
                "annotation, diagnosis unless the definition requires it, or any other listed factor. "
                "Vary wording, age, context, tone, spelling, and explicitness. Keep content non-graphic; "
                "never give quantities or operational self-harm instructions. Return exactly one JSON "
                f"array of {remaining} strings and no commentary.\nRequested factor: {label}\n"
                f"Meaning:\n{definitions}"
            )
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False,
                add_generation_prompt=True,
            )
            encoded = tokenizer(rendered, return_tensors="pt", truncation=True,
                                max_length=480).to("cuda")
            torch.manual_seed(SEED + label_index * 2 + attempt)
            generated = model.generate(
                **encoded, max_new_tokens=1400, do_sample=True, temperature=.78,
                top_p=.92, repetition_penalty=1.06,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            raw = tokenizer.decode(
                generated[0, encoded.input_ids.shape[1]:], skip_special_tokens=True,
            )
            clean = _clean_examples(_parse_array(raw), label, seen)
            accepted.extend(clean[:remaining])
            attempt_audit.append({"attempt": attempt + 1, "accepted": len(clean), "raw": raw})
        examples.append(accepted[:PER_LABEL]); audits.append(attempt_audit)
        print(f"Factor V6 generation {label_index + 1}/24 {label}: "
              f"accepted={len(examples[-1])}", flush=True)
    payload = {
        "training_version": TRAINING_VERSION, "model": "Qwen/Qwen2.5-7B-Instruct",
        "examples_per_label_requested": PER_LABEL, "examples": examples,
        "counts": [len(items) for items in examples], "audits": audits,
        "peak_memory_gb": float(torch.cuda.max_memory_allocated() / 1024 ** 3),
    }
    SYNTHETIC_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items()
                      if key not in ("examples", "audits")}, indent=2), flush=True)
    del model; torch.cuda.empty_cache()
    return payload


def _synthetic_rows():
    saved = json.loads(SYNTHETIC_FILE.read_text(encoding="utf-8"))
    texts, labels = [], []
    for label, items in enumerate(saved["examples"]):
        for text in items:
            texts.append(str(text)); labels.append(label)
    return texts, np.asarray(labels, dtype=np.int64)


def _fit_predict(train_texts, train_targets, valid_texts, synthetic_texts,
                 synthetic_labels, save_path=None):
    combined = list(train_texts) + list(synthetic_texts)
    vectorizer = _vectorizer(); matrix = vectorizer.fit_transform(combined)
    real_count = len(train_texts); real_matrix = matrix[:real_count]
    synthetic_matrix = matrix[real_count:]
    valid_matrix = (vectorizer.transform(list(valid_texts))
                    if len(valid_texts) else None)
    probabilities = np.zeros((len(valid_texts), config.NUM_FACTORS), dtype=np.float32)
    models = []
    for label in range(config.NUM_FACTORS):
        synthetic_target = (synthetic_labels == label).astype(np.int8)
        target = np.concatenate((train_targets[:, label], synthetic_target))
        sample_weight = np.concatenate((
            np.ones(real_count, dtype=np.float32),
            np.where(synthetic_target > 0, POSITIVE_SYNTHETIC_WEIGHT,
                     NEGATIVE_SYNTHETIC_WEIGHT).astype(np.float32),
        ))
        model = LogisticRegression(
            C=2., class_weight="balanced", max_iter=600, solver="liblinear",
            random_state=SEED + label,
        )
        model.fit(matrix, target, sample_weight=sample_weight)
        if valid_matrix is not None:
            probabilities[:, label] = model.predict_proba(valid_matrix)[:, 1]
        models.append(model)
    if save_path is not None:
        joblib.dump({
            "training_version": TRAINING_VERSION, "vectorizer": vectorizer,
            "models": models, "synthetic_count": len(synthetic_texts),
        }, save_path)
    return probabilities


def _current_v3_probability():
    calibration = json.loads((config.OUTPUT_DIR / "factor_cross_encoder_v2" /
                              "calibration.json").read_text(encoding="utf-8"))
    base_saved = np.load(config.OUTPUT_DIR / "factor_cv" / "factor_oof_predictions.npz")
    base = (.7 * base_saved["semantic"] + .3 * base_saved["cpu"])
    old = np.load(config.OUTPUT_DIR / "factor_cross_encoder" /
                  "oof_predictions.npz")["probabilities"]
    new = np.load(config.OUTPUT_DIR / "factor_cross_encoder_v2" /
                  "oof_predictions.npz")["probabilities"]
    probability = (float(calibration["base_weight"]) * base
                   + float(calibration["old_cross_weight"]) * old
                   + float(calibration["new_cross_weight"]) * new)
    return probability.astype(np.float32), calibration


def _select_label_weights(base, lexical, targets, indices, prevalence):
    weights = np.zeros(config.NUM_FACTORS, dtype=np.float32)
    for label in range(config.NUM_FACTORS):
        count = max(1, int(round(len(indices) * prevalence[label] * 1.10)))
        truth = targets[indices, label]
        best = (-1., 0.)
        for weight in WEIGHT_GRID:
            score = (1. - weight) * base[indices, label] + weight * lexical[indices, label]
            selected = np.argpartition(score, len(score) - min(count, len(score)))[-min(count, len(score)):]
            prediction = np.zeros(len(score), dtype=bool); prediction[selected] = True
            value = f1_score(truth, prediction, zero_division=0)
            best = max(best, (float(value), -float(weight)))
        weights[label] = -best[1]
    return weights


def cross_validate():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame = load_train_data().reset_index(drop=True)
    targets = np.vstack(frame.factor_vector.to_numpy()).astype(np.int8)
    texts = frame.text.astype(str).tolist(); synthetic_texts, synthetic_labels = _synthetic_rows()
    folds = list(StratifiedGroupKFold(
        n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED,
    ).split(np.zeros(len(frame)), frame.risk_label, frame.anon_user_id))
    lexical = np.zeros_like(targets, dtype=np.float32)
    for fold, (train_idx, valid_idx) in enumerate(folds):
        print(f"Factor V6 lexical fold {fold}/4", flush=True)
        lexical[valid_idx] = _fit_predict(
            frame.text.iloc[train_idx].astype(str).tolist(), targets[train_idx],
            frame.text.iloc[valid_idx].astype(str).tolist(),
            synthetic_texts, synthetic_labels,
        )
    base, old_calibration = _current_v3_probability()
    crossfit_prediction = np.zeros_like(targets, dtype=bool)
    baseline_prediction = np.zeros_like(targets, dtype=bool)
    fold_rows = []; selected_weights = []
    for fold, (fit_idx, valid_idx) in enumerate(folds):
        prevalence = targets[fit_idx].mean(0)
        weights = _select_label_weights(base, lexical, targets, fit_idx, prevalence)
        candidate = (1. - weights) * base[valid_idx] + weights * lexical[valid_idx]
        crossfit_prediction[valid_idx] = _rank_decode(candidate, prevalence, 1.10)
        baseline_prediction[valid_idx] = _rank_decode(base[valid_idx], prevalence, 1.10)
        fold_rows.append({
            "fold": fold,
            "baseline_macro_f1": float(f1_score(
                targets[valid_idx], baseline_prediction[valid_idx], average="macro", zero_division=0)),
            "candidate_macro_f1": float(f1_score(
                targets[valid_idx], crossfit_prediction[valid_idx], average="macro", zero_division=0)),
            "weights": weights.tolist(),
        })
        selected_weights.append(weights)
    baseline_crossfit = float(f1_score(targets, baseline_prediction, average="macro", zero_division=0))
    candidate_crossfit = float(f1_score(targets, crossfit_prediction, average="macro", zero_division=0))
    production_weights = np.median(np.vstack(selected_weights), axis=0).astype(np.float32)
    prevalence = targets.mean(0)
    production_probability = (1. - production_weights) * base + production_weights * lexical
    baseline_production_prediction = _rank_decode(base, prevalence, 1.10)
    production_prediction = _rank_decode(production_probability, prevalence, 1.10)
    baseline_production = float(f1_score(
        targets, baseline_production_prediction, average="macro", zero_division=0))
    production_score = float(f1_score(
        targets, production_prediction, average="macro", zero_division=0))
    per_label = []
    for label, name in enumerate(config.FACTOR_LABELS):
        per_label.append({
            "label": name, "support": int(targets[:, label].sum()),
            "weight": float(production_weights[label]),
            "baseline_f1": float(f1_score(targets[:, label], baseline_production_prediction[:, label], zero_division=0)),
            "candidate_f1": float(f1_score(targets[:, label], production_prediction[:, label], zero_division=0)),
        })
    groups = frame.anon_user_id.astype(str).to_numpy(); users = np.unique(groups)
    rng = np.random.default_rng(SEED); deltas = []
    for _ in range(2000):
        sample = rng.choice(users, size=len(users), replace=True)
        positions = np.concatenate([np.flatnonzero(groups == user) for user in sample])
        old = f1_score(targets[positions], baseline_prediction[positions], average="macro", zero_division=0)
        new = f1_score(targets[positions], crossfit_prediction[positions], average="macro", zero_division=0)
        deltas.append(float(new - old))
    deltas = np.asarray(deltas)
    bootstrap = {
        "mean_delta": float(deltas.mean()), "p05_delta": float(np.quantile(deltas, .05)),
        "p95_delta": float(np.quantile(deltas, .95)),
        "positive_fraction": float((deltas > 0).mean()),
    }
    adopted = bool(candidate_crossfit >= baseline_crossfit + .003
                   and production_score >= baseline_production + .003
                   and bootstrap["positive_fraction"] >= .8)
    payload = {
        "training_version": TRAINING_VERSION, "folds": fold_rows,
        "baseline_crossfit_macro_f1": baseline_crossfit,
        "candidate_crossfit_macro_f1": candidate_crossfit,
        "baseline_production_oof_macro_f1": baseline_production,
        "candidate_production_oof_macro_f1": production_score,
        "production_weights": production_weights.tolist(), "per_label": per_label,
        "user_cluster_bootstrap": bootstrap, "adopted": adopted,
    }
    RESULTS_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    CALIBRATION_FILE.write_text(json.dumps({
        "training_version": TRAINING_VERSION, "adopted": adopted,
        "weights": production_weights.tolist(), "prevalence_ratio": 1.10,
        "training_prevalence": prevalence.tolist(),
    }, indent=2), encoding="utf-8")
    np.savez_compressed(OOF_FILE, probabilities=lexical, targets=targets)
    print(json.dumps(payload, indent=2), flush=True)
    return payload


def train_full():
    calibration = json.loads(CALIBRATION_FILE.read_text(encoding="utf-8"))
    if not calibration.get("adopted", False):
        raise RuntimeError("Factor V6 did not pass strict cross-fit adoption")
    frame = load_train_data().reset_index(drop=True)
    targets = np.vstack(frame.factor_vector.to_numpy()).astype(np.int8)
    synthetic_texts, synthetic_labels = _synthetic_rows()
    _fit_predict(frame.text.astype(str).tolist(), targets, [],
                 synthetic_texts, synthetic_labels, save_path=FULL_MODEL_FILE)
    print(f"Factor V6 full model ready: {FULL_MODEL_FILE}", flush=True)
    return FULL_MODEL_FILE


if __name__ == "__main__":
    cross_validate()
