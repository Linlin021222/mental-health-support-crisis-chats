"""Five-fold user-disjoint definition-guided sentence evidence for Task 2.

V17 showed a useful +0.00235 on one outer fold, but its semantic-bank teacher
existed only for fold 0.  V27 starts every fold from that fold's accepted V3
prototype cross-encoder, mines evidence only from its training users, and
continues the model on sentence-level positives plus boundary-aware confusing
negatives.  The decoder ratio and replacement weight are fixed in advance.
"""
from __future__ import annotations

import json

import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from transformers import AutoTokenizer

from configs.config import config
from inference.factor_nli import _rank_decode
from preprocess.preprocess import load_train_data
import trainer.factor_sentence_evidence_v16 as sentence
from trainer.factor_llm_lexical_v6 import _current_v3_probability
from utils.seed import seed_everything


OUTPUT = config.OUTPUT_DIR / "factor_sentence_evidence_cv_v27"
OOF_FILE = OUTPUT / "oof_predictions.npz"
RESULTS = OUTPUT / "cv_results.json"
TRAINING_VERSION = "five-fold-definition-sentence-evidence-v27"
ACCEPTED_DIR = config.OUTPUT_DIR / "factor_cross_encoder_v2"
REPLACEMENT_WEIGHT = 0.20
TOPK_RATIO = 1.10
MIN_EVIDENCE_SCORE = 0.20


def _paths(fold):
    return (
        OUTPUT / f"fold{fold}_model.pt",
        OUTPUT / f"fold{fold}_valid.npz",
        OUTPUT / f"fold{fold}_pseudo_evidence.jsonl",
    )


def _train_fold(fold, train_idx, valid_idx, frame, targets, counts, tokenizer, device):
    checkpoint, predictions_file, evidence_file = _paths(fold)
    if predictions_file.exists() and evidence_file.exists():
        saved = np.load(predictions_file)
        if (str(saved["training_version"]) == TRAINING_VERSION
                and np.array_equal(saved["valid_indices"], valid_idx)):
            summary = json.loads(str(saved["summary"]))
            print(f"V27 fold {fold}: resumed", flush=True)
            return saved["probabilities"].astype(np.float32), summary

    source = ACCEPTED_DIR / f"fold{fold}_model.pt"
    if not source.exists():
        raise FileNotFoundError(f"Missing accepted V3 checkpoint: {source}")
    seed_everything(config.SEED + 2700 + fold)
    model = sentence._load_model(source, device)

    # Reuse the audited V17 mining/training implementation, with a strict
    # confidence floor and the richer boundary-aware semantic prompt bank.
    old_minimum = sentence.MIN_EVIDENCE_SCORE
    sentence.MIN_EVIDENCE_SCORE = MIN_EVIDENCE_SCORE
    try:
        evidence_records, pairs = sentence._mine_evidence(
            frame, train_idx, targets, counts, model, tokenizer, device,
        )
        evidence_file.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in evidence_records) + "\n",
            encoding="utf-8",
        )
        print(
            f"V27 fold {fold}: positive post-label records={len(evidence_records)}, "
            f"sentence pairs={len(pairs)}", flush=True,
        )
        train_loss = sentence._train(model, tokenizer, pairs, device)
        probability = sentence._predict_posts(
            model, tokenizer,
            frame.text.iloc[valid_idx].astype(str).tolist(), device,
        )
    finally:
        sentence.MIN_EVIDENCE_SCORE = old_minimum

    selected = np.asarray(
        [row["selected_sentence_count"] for row in evidence_records], dtype=np.int64
    )
    summary = {
        "fold": fold,
        "training_version": TRAINING_VERSION,
        "train_rows": int(len(train_idx)),
        "valid_rows": int(len(valid_idx)),
        "pair_count": int(len(pairs)),
        "positive_post_label_records": int(len(evidence_records)),
        "selected_sentences": int(selected.sum()),
        "zero_sentence_records": int((selected == 0).sum()),
        "minimum_semantic_score": MIN_EVIDENCE_SCORE,
        "train_loss": float(train_loss),
        "initialised_from": str(source),
    }
    torch.save(model.state_dict(), checkpoint)
    np.savez_compressed(
        predictions_file,
        probabilities=probability,
        valid_indices=valid_idx,
        summary=json.dumps(summary),
        training_version=TRAINING_VERSION,
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return probability, summary


def _user_bootstrap(targets, baseline, candidate, groups, seed=272727, draws=2000):
    unique = np.unique(groups)
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(draws):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([np.flatnonzero(groups == group) for group in sampled])
        old = f1_score(
            targets[indices], baseline[indices], average="macro", zero_division=0
        )
        new = f1_score(
            targets[indices], candidate[indices], average="macro", zero_division=0
        )
        deltas.append(new - old)
    values = np.asarray(deltas, dtype=np.float64)
    return {
        "mean_delta": float(values.mean()),
        "p05_delta": float(np.quantile(values, .05)),
        "p95_delta": float(np.quantile(values, .95)),
        "positive_fraction": float((values > 0).mean()),
    }


def cross_validate(only_fold0=False):
    if not torch.cuda.is_available():
        raise RuntimeError("Factor sentence evidence V27 requires CUDA")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame = load_train_data().reset_index(drop=True)
    targets = np.vstack(frame.factor_vector.to_numpy()).astype(np.int8)
    counts = np.vstack(frame.factor_counts.to_numpy()).astype(np.float32)
    risk = frame.risk_label.to_numpy(dtype=np.int64)
    groups = frame.anon_user_id.astype(str).to_numpy()
    folds = list(StratifiedGroupKFold(
        n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED,
    ).split(np.zeros(len(frame)), risk, groups))
    tokenizer = AutoTokenizer.from_pretrained(
        config.FACTOR_NLI_MODEL_NAME, use_fast=True, local_files_only=True,
    )
    device = torch.device("cuda")
    sentence_oof = np.zeros_like(targets, dtype=np.float32)
    summaries = []
    selected_folds = folds[:1] if only_fold0 else folds
    for fold, (train_idx, valid_idx) in enumerate(selected_folds):
        probability, summary = _train_fold(
            fold, train_idx, valid_idx, frame, targets, counts, tokenizer, device
        )
        sentence_oof[valid_idx] = probability
        summaries.append(summary)

    current, calibration = _current_v3_probability()
    accepted = np.load(ACCEPTED_DIR / "oof_predictions.npz")["probabilities"].astype(np.float32)
    component_weight = float(calibration["new_cross_weight"])
    indices = folds[0][1] if only_fold0 else np.arange(len(frame))
    candidate_probability = (
        current[indices]
        + component_weight * REPLACEMENT_WEIGHT
        * (sentence_oof[indices] - accepted[indices])
    )
    if only_fold0:
        prevalence = targets[folds[0][0]].mean(0)
    else:
        prevalence = targets.mean(0)
    baseline_prediction = _rank_decode(current[indices], prevalence, TOPK_RATIO)
    candidate_prediction = _rank_decode(candidate_probability, prevalence, TOPK_RATIO)
    standalone_prediction = _rank_decode(sentence_oof[indices], prevalence, TOPK_RATIO)
    truth = targets[indices]
    baseline = float(f1_score(
        truth, baseline_prediction, average="macro", zero_division=0
    ))
    candidate = float(f1_score(
        truth, candidate_prediction, average="macro", zero_division=0
    ))
    standalone = float(f1_score(
        truth, standalone_prediction, average="macro", zero_division=0
    ))
    per_label = [{
        "label": config.ID2FACTOR[label],
        "support": int(truth[:, label].sum()),
        "baseline_f1": float(f1_score(
            truth[:, label], baseline_prediction[:, label], zero_division=0
        )),
        "candidate_f1": float(f1_score(
            truth[:, label], candidate_prediction[:, label], zero_division=0
        )),
    } for label in range(config.NUM_FACTORS)]

    bootstrap = _user_bootstrap(
        truth, baseline_prediction, candidate_prediction, groups[indices]
    )
    adopted = bool(
        not only_fold0
        and candidate >= baseline + .005
        and bootstrap["positive_fraction"] >= .80
        and bootstrap["p05_delta"] >= 0
    )
    payload = {
        "training_version": TRAINING_VERSION,
        "evaluation": (
            "strict outer fold0" if only_fold0
            else "five-fold user-disjoint OOF; fixed replacement and decoder"
        ),
        "fixed_policy": {
            "prototype_component_replacement": REPLACEMENT_WEIGHT,
            "effective_total_weight": component_weight * REPLACEMENT_WEIGHT,
            "prevalence_ratio": TOPK_RATIO,
            "minimum_semantic_score": MIN_EVIDENCE_SCORE,
        },
        "folds": summaries,
        "baseline_macro_f1": baseline,
        "sentence_model_standalone_macro_f1": standalone,
        "candidate_macro_f1": candidate,
        "delta": candidate - baseline,
        "user_cluster_bootstrap": bootstrap,
        "per_label": per_label,
        "promising_for_full_oof": bool(candidate >= baseline + .005),
        "adopted": adopted,
    }
    target = OUTPUT / ("fold0_results.json" if only_fold0 else "cv_results.json")
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if not only_fold0:
        np.savez_compressed(
            OOF_FILE, probabilities=sentence_oof, targets=targets,
            training_version=TRAINING_VERSION,
        )
    print(json.dumps(payload, indent=2), flush=True)
    return payload


if __name__ == "__main__":
    cross_validate()
