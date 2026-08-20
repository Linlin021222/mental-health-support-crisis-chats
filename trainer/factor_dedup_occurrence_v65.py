"""Content-deduplicated, occurrence-aware sparse specialist for Task 2.

The submitted target is binary, while repeated factor names encode weak
mention salience.  This expert keeps binary targets but uses capped log-count
sample weights.  Exact normalised duplicate posts are inversely weighted so a
copied timeline entry cannot dominate a label classifier.  Rare labels also
receive two low-weight taxonomy descriptions as semantic positive anchors.
"""
from __future__ import annotations

import json
import re

import joblib
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

from baseline import _factor_probabilities, _vectorizer
from configs.config import config
from inference.factor_nli import _rank_decode
from preprocess.preprocess import load_train_data
from trainer.factor_balanced_sparse_v48 import _current_components, _ensemble
from trainer.factor_sentence_evidence_cv_v27 import _user_bootstrap


OUTPUT = config.OUTPUT_DIR / "factor_dedup_occurrence_v65"
OOF_FILE = OUTPUT / "oof_predictions.npz"
RESULTS = OUTPUT / "results.json"
CALIBRATION = OUTPUT / "calibration.json"
TRAINING_VERSION = "dedup-occurrence-sparse-v65"
OCCURRENCE_ALPHA = .15
MAX_OCCURRENCE_BOOST = 1.50
RARE_SUPPORT = 60
SEMANTIC_ANCHOR_WEIGHT = .25


def _normalise(text):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", str(text).casefold())).strip()


def _cluster_weights(texts):
    keys = np.asarray([_normalise(text) for text in texts])
    unique, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
    del unique
    return (1.0 / counts[inverse]).astype(np.float32)


def _fit_models(vectorizer, matrix, targets, counts, texts):
    models = []
    cluster = _cluster_weights(texts)
    descriptions = list(config.FACTOR_DESCRIPTIONS) + list(config.FACTOR_NLI_HYPOTHESES)
    description_matrix = vectorizer.transform(descriptions)
    for label in range(config.NUM_FACTORS):
        target = targets[:, label].astype(np.int8)
        if len(np.unique(target)) < 2:
            model = DummyClassifier(strategy="constant", constant=int(target[0]))
            model.fit(matrix, target); models.append(model); continue
        repeats = np.maximum(counts[:, label] - 1.0, 0.0)
        boost = np.minimum(
            1.0 + OCCURRENCE_ALPHA * np.log1p(repeats), MAX_OCCURRENCE_BOOST,
        )
        weights = cluster * np.where(target > 0, boost, 1.0)
        fit_matrix, fit_target, fit_weight = matrix, target, weights
        if int(target.sum()) < RARE_SUPPORT:
            # Two descriptions of the same label are deliberately weak
            # anchors, not synthetic Reddit posts.  Other label definitions
            # are excluded to avoid flooding an 8-positive label with noise.
            anchor_rows = np.asarray([label, config.NUM_FACTORS + label])
            fit_matrix = __import__("scipy").sparse.vstack(
                [matrix, description_matrix[anchor_rows]], format="csr",
            )
            fit_target = np.concatenate([target, np.ones(2, dtype=np.int8)])
            fit_weight = np.concatenate([
                weights, np.full(2, SEMANTIC_ANCHOR_WEIGHT, dtype=np.float32),
            ])
        model = LogisticRegression(
            C=2.0, class_weight="balanced", max_iter=600,
            solver="liblinear", random_state=config.SEED + 6500 + label,
        )
        model.fit(fit_matrix, fit_target, sample_weight=fit_weight)
        models.append(model)
    return models


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame = load_train_data().reset_index(drop=True)
    texts = frame.text.astype(str).to_numpy()
    targets = np.vstack(frame.factor_vector.to_numpy()).astype(np.int8)
    counts = np.vstack(frame.factor_counts.to_numpy()).astype(np.float32)
    groups = frame.anon_user_id.astype(str).to_numpy()
    risk = frame.risk_label.to_numpy(dtype=np.int64)
    folds = list(StratifiedGroupKFold(
        n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED,
    ).split(np.zeros(len(frame)), risk, groups))
    oof = np.zeros_like(targets, dtype=np.float32)
    for fold, (fit, valid) in enumerate(folds):
        print(f"V65 sparse fold {fold}: train={len(fit)} valid={len(valid)}", flush=True)
        vectorizer = _vectorizer()
        # Taxonomy text is public task information and contains no held-out
        # labels. Including it in the vocabulary is therefore leak-free.
        vocabulary_corpus = list(texts[fit]) + list(config.FACTOR_DESCRIPTIONS) \
            + list(config.FACTOR_NLI_HYPOTHESES)
        vectorizer.fit(vocabulary_corpus)
        train_matrix = vectorizer.transform(texts[fit])
        valid_matrix = vectorizer.transform(texts[valid])
        models = _fit_models(
            vectorizer, train_matrix, targets[fit], counts[fit], texts[fit],
        )
        oof[valid] = _factor_probabilities(models, valid_matrix)
        joblib.dump({
            "training_version": TRAINING_VERSION, "fold": fold,
            "vectorizer": vectorizer, "models": models,
        }, OUTPUT / f"fold{fold}.joblib", compress=3)
    semantic, old_cpu, old_cross, prototype, calibration = _current_components()
    baseline_probability = _ensemble(
        semantic, old_cpu, old_cross, prototype, calibration,
    )
    candidate_probability = _ensemble(
        semantic, oof, old_cross, prototype, calibration,
    )
    baseline = np.zeros_like(targets, dtype=bool)
    candidate = np.zeros_like(targets, dtype=bool)
    for fit, valid in folds:
        prevalence = targets[fit].mean(0)
        baseline[valid] = _rank_decode(baseline_probability[valid], prevalence, 1.10)
        candidate[valid] = _rank_decode(candidate_probability[valid], prevalence, 1.10)
    old_score = float(f1_score(targets, baseline, average="macro", zero_division=0))
    new_score = float(f1_score(targets, candidate, average="macro", zero_division=0))
    bootstrap = _user_bootstrap(
        targets, baseline, candidate, groups, seed=656565, draws=4000,
    )
    per_label = []
    for label in range(config.NUM_FACTORS):
        old = float(f1_score(targets[:, label], baseline[:, label], zero_division=0))
        new = float(f1_score(targets[:, label], candidate[:, label], zero_division=0))
        per_label.append({
            "label": config.ID2FACTOR[label], "support": int(targets[:, label].sum()),
            "baseline_f1": old, "candidate_f1": new, "delta": new-old,
        })
    adopted = bool(
        new_score >= old_score + .004
        and bootstrap["positive_fraction"] >= .80
        and bootstrap["p05_delta"] >= 0
    )
    payload = {
        "training_version": TRAINING_VERSION,
        "evaluation": "five original user-disjoint OOF folds; fixed production ensemble and decoder",
        "method": {
            "duplicate_weight": "1 / exact-normalised cluster size",
            "occurrence_alpha": OCCURRENCE_ALPHA,
            "maximum_occurrence_boost": MAX_OCCURRENCE_BOOST,
            "rare_semantic_anchor_weight": SEMANTIC_ANCHOR_WEIGHT,
        },
        "baseline_macro_f1": old_score, "candidate_macro_f1": new_score,
        "delta": new_score-old_score, "user_cluster_bootstrap": bootstrap,
        "per_label": per_label, "adopted": adopted,
    }
    np.savez_compressed(OOF_FILE, probabilities=oof, targets=targets)
    RESULTS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    CALIBRATION.write_text(json.dumps({
        "training_version": TRAINING_VERSION, "adopted": adopted,
        "baseline_macro_f1": old_score, "candidate_macro_f1": new_score,
        "bootstrap": bootstrap,
    }, indent=2), encoding="utf-8")
    print(json.dumps({k:v for k,v in payload.items() if k != "per_label"}, indent=2), flush=True)
    return payload


if __name__ == "__main__":
    main()
