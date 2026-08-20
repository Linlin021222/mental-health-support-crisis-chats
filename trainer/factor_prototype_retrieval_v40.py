"""Strict fold-0 prototype cross-encoder retrieval/aggregation ablation.

The experiment modifies the strongest accepted Task-2 component itself:
  * prototype reliability is learned only from the outer training partition;
  * every label hypothesis receives a cross-user retrieved positive example;
  * the nearest taxonomy confusion is stated as an explicit exclusion.

No label or prevalence information from the outer validation users is used to
construct hypotheses or aggregation weights.
"""
from __future__ import annotations

import json
import math
import re

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import FeatureUnion
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from configs.config import config
from inference.factor_nli import _entailment_index, _rank_decode
from preprocess.factor_prototypes import FACTOR_PROTOTYPES
from preprocess.preprocess import load_train_data
from trainer.factor_cross_encoder_v2 import CONFUSION_GROUPS
from trainer.factor_mhlat_v4 import _current_components


OUTPUT = config.OUTPUT_DIR / "factor_prototype_retrieval_v40"
RESULTS = OUTPUT / "fold0_results.json"
PREDICTIONS = OUTPUT / "fold0_valid.npz"
CHECKPOINT = config.OUTPUT_DIR / "factor_cross_encoder_v2" / "fold0_model.pt"
CURRENT_PROTOTYPE = config.OUTPUT_DIR / "factor_cross_encoder_v2" / "oof_predictions.npz"
TRAINING_VERSION = "prototype-retrieval-boundary-v40"
MAX_EXAMPLE_WORDS = 42


def _vectorizer():
    return FeatureUnion([
        ("word", TfidfVectorizer(
            lowercase=True, ngram_range=(1, 2), min_df=2, max_features=45000,
            sublinear_tf=True, strip_accents="unicode",
        )),
        ("char", TfidfVectorizer(
            analyzer="char_wb", lowercase=True, ngram_range=(3, 5), min_df=2,
            max_features=35000, sublinear_tf=True,
        )),
    ])


def _confusion_label(label, targets):
    candidates = set()
    for group in CONFUSION_GROUPS:
        if label in group:
            candidates.update(item for item in group if item != label)
    if not candidates:
        return (label + 1) % config.NUM_FACTORS
    # Prefer a genuinely common boundary, estimated only on outer-train labels.
    positive = targets[:, label] > 0
    return max(candidates, key=lambda item: int(targets[positive, item].sum()))


def _prototype_weights(vectorizer, train_matrix, targets):
    flat = [prototype for group in FACTOR_PROTOTYPES for prototype in group]
    prototype_matrix = vectorizer.transform(flat)
    weights, diagnostics, offset = [], [], 0
    for label, prototypes in enumerate(FACTOR_PROTOTYPES):
        values = []
        truth = targets[:, label]
        for prototype in prototypes:
            similarity = (train_matrix @ prototype_matrix[offset].T).toarray().ravel()
            offset += 1
            auc = roc_auc_score(truth, similarity) if 0 < truth.sum() < len(truth) else .5
            values.append(float(auc))
        # Shrink strongly toward the robust old mean. A prototype must show
        # train-only discriminability before receiving materially more weight.
        centred = (np.asarray(values) - np.mean(values)) / .12
        reliability = np.exp(np.clip(centred, -2., 2.))
        reliability = .35 + .65 * reliability / max(reliability.mean(), 1e-8)
        reliability /= reliability.sum()
        weights.append(reliability.astype(np.float32))
        diagnostics.append({"label": config.ID2FACTOR[label], "auc": values,
                            "weights": reliability.tolist()})
    return weights, diagnostics


def _best_excerpt(reference, query, prototype):
    sentences = [item.strip() for item in re.split(r"(?<=[.!?])\s+|\n+", str(reference))
                 if item.strip()]
    if not sentences:
        sentences = [str(reference)]
    signal = set(re.findall(r"[a-z']{3,}", (str(query) + " " + prototype).lower()))
    def score(sentence):
        words = set(re.findall(r"[a-z']{3,}", sentence.lower()))
        return len(signal & words) / math.sqrt(max(1, len(words)))
    excerpt = max(sentences, key=score)
    words = excerpt.split()
    return " ".join(words[:MAX_EXAMPLE_WORDS])


def _retrieval_examples(train_texts, valid_texts, train_targets, train_matrix, valid_matrix):
    examples = [[""] * config.NUM_FACTORS for _ in valid_texts]
    for label in range(config.NUM_FACTORS):
        positives = np.flatnonzero(train_targets[:, label] > 0)
        if not len(positives):
            continue
        similarity = valid_matrix @ train_matrix[positives].T
        chosen = positives[np.asarray(similarity.argmax(axis=1)).ravel()]
        prototype = FACTOR_PROTOTYPES[label][0]
        for row, reference in enumerate(chosen):
            examples[row][label] = _best_excerpt(
                train_texts[int(reference)], valid_texts[row], prototype,
            )
    return examples


def _guided_hypothesis(label, prototype, example, confusion):
    boundary = FACTOR_PROTOTYPES[confusion][0]
    return (
        f"{prototype} Label-grounding example from another annotated author: "
        f"\"{example}\" This label should not be inferred when the post only "
        f"supports the distinct concept: {boundary}"
    )


@torch.no_grad()
def _predict(model, tokenizer, texts, examples, weights, confusion, device):
    model.eval(); entailment = _entailment_index(model)
    result = np.zeros((len(texts), config.NUM_FACTORS), dtype=np.float32)
    batch_size = max(1, config.FACTOR_CROSS_ENCODER_BATCH_SIZE)
    for label, prototypes in enumerate(tqdm(FACTOR_PROTOTYPES, desc="V40 guided labels")):
        scores = np.zeros((len(texts), len(prototypes)), dtype=np.float32)
        for prototype_index, prototype in enumerate(prototypes):
            for start in range(0, len(texts), batch_size):
                batch = texts[start:start + batch_size]
                hypotheses = [
                    _guided_hypothesis(
                        label, prototype, examples[start + local][label], confusion[label]
                    ) for local in range(len(batch))
                ]
                encoded = tokenizer(
                    batch, hypotheses, padding=True, truncation="only_first",
                    max_length=config.FACTOR_NLI_MAX_LENGTH,
                    stride=min(128, config.FACTOR_NLI_MAX_LENGTH // 3),
                    return_overflowing_tokens=True, return_tensors="pt",
                )
                mapping = encoded.pop("overflow_to_sample_mapping").cpu().numpy()
                selected = []
                for local in range(len(batch)):
                    indices = np.flatnonzero(mapping == local)
                    if len(indices) > config.FACTOR_NLI_MAX_CHUNKS:
                        positions = np.linspace(0, len(indices)-1, config.FACTOR_NLI_MAX_CHUNKS).round().astype(int)
                        indices = indices[positions]
                    selected.extend(indices.tolist())
                selected = np.asarray(selected, dtype=np.int64)
                selected_mapping = mapping[selected]
                take = torch.tensor(selected, dtype=torch.long)
                encoded = {key: value.index_select(0, take).to(device)
                           for key, value in encoded.items()}
                with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    logits = model(**encoded).logits.float()
                values = torch.softmax(logits, -1)[:, entailment].cpu().numpy()
                for local in range(len(batch)):
                    scores[start + local, prototype_index] = values[selected_mapping == local].max()
        result[:, label] = scores @ weights[label]
    return result


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("V40 requires CUDA")
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Missing accepted prototype checkpoint: {CHECKPOINT}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame = load_train_data().reset_index(drop=True)
    targets = np.vstack(frame.factor_vector.to_numpy()).astype(np.int8)
    groups = frame.anon_user_id.astype(str).to_numpy()
    folds = list(StratifiedGroupKFold(
        n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED,
    ).split(np.zeros(len(frame)), frame.risk_label, groups))
    train_idx, valid_idx = folds[0]
    train_texts = frame.text.iloc[train_idx].astype(str).tolist()
    valid_texts = frame.text.iloc[valid_idx].astype(str).tolist()
    vectorizer = _vectorizer(); train_matrix = vectorizer.fit_transform(train_texts)
    valid_matrix = vectorizer.transform(valid_texts)
    weights, weight_diagnostics = _prototype_weights(
        vectorizer, train_matrix, targets[train_idx],
    )
    confusion = [_confusion_label(label, targets[train_idx])
                 for label in range(config.NUM_FACTORS)]
    examples = _retrieval_examples(
        train_texts, valid_texts, targets[train_idx], train_matrix, valid_matrix,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        config.FACTOR_NLI_MODEL_NAME, use_fast=True, local_files_only=True,
    )
    device = torch.device(config.DEVICE)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.FACTOR_NLI_MODEL_NAME, dtype=torch.float32, local_files_only=True,
    ).to(device)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    candidate = _predict(model, tokenizer, valid_texts, examples, weights, confusion, device)

    current, all_targets, calibration = _current_components()
    accepted = np.load(CURRENT_PROTOTYPE)["probabilities"].astype(np.float32)
    prevalence = all_targets[train_idx].mean(0)
    ratio = float(calibration["prevalence_ratio"])
    baseline_prediction = _rank_decode(current[valid_idx], prevalence, ratio)
    baseline = float(f1_score(all_targets[valid_idx], baseline_prediction,
                              average="macro", zero_division=0))
    standalone = float(f1_score(
        all_targets[valid_idx], _rank_decode(candidate, prevalence, ratio),
        average="macro", zero_division=0,
    ))
    grid = []
    for replacement in (0., .10, .20, .25, .35, .50, .75, 1.0):
        prototype = (1-replacement)*accepted[valid_idx] + replacement*candidate
        probability = (
            float(calibration["base_weight"]) * (
                (current[valid_idx]
                 - float(calibration["old_cross_weight"]) * np.load(
                     config.OUTPUT_DIR / "factor_cross_encoder" / "oof_predictions.npz"
                 )["probabilities"][valid_idx]
                 - float(calibration["new_cross_weight"]) * accepted[valid_idx])
                / float(calibration["base_weight"])
            )
            + float(calibration["old_cross_weight"]) * np.load(
                config.OUTPUT_DIR / "factor_cross_encoder" / "oof_predictions.npz"
            )["probabilities"][valid_idx]
            + float(calibration["new_cross_weight"]) * prototype
        )
        score = float(f1_score(
            all_targets[valid_idx], _rank_decode(probability, prevalence, ratio),
            average="macro", zero_division=0,
        ))
        grid.append({"replacement": replacement, "macro_f1": score,
                     "delta": score-baseline})
    fixed = next(item for item in grid if item["replacement"] == .25)
    payload = {
        "training_version": TRAINING_VERSION,
        "evaluation_scope": "strict user-disjoint fold0",
        "current_baseline_macro_f1": baseline,
        "guided_standalone_macro_f1": standalone,
        "fixed_25pct_replacement": fixed,
        "grid": sorted(grid, key=lambda item: item["macro_f1"], reverse=True),
        "prototype_weight_diagnostics": weight_diagnostics,
        "confusion_labels": [config.ID2FACTOR[item] for item in confusion],
        "promising": bool(fixed["delta"] >= .003),
    }
    np.savez_compressed(PREDICTIONS, valid_indices=valid_idx, probabilities=candidate)
    RESULTS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items()
                      if key not in ("prototype_weight_diagnostics", "confusion_labels")},
                     indent=2), flush=True)
    return payload


if __name__ == "__main__":
    main()
