"""Nested, label-selective fusion of the leak-free V25 definition ranker."""
from __future__ import annotations

import json

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from configs.config import config
from inference.factor_nli import _rank_decode
from preprocess.preprocess import load_train_data
from trainer.factor_definition_oof_ranker_v25 import FEATURES, TRAINING_VERSION as FEATURE_VERSION
from trainer.factor_definition_rank_fusion_v24 import _inner_oof, _percentile_rank, _label_f1
from trainer.factor_definition_ranker_v23 import _fit_rankers
from trainer.factor_llm_lexical_v6 import _current_v3_probability


OUTPUT = config.OUTPUT_DIR / "factor_definition_selective_v26"
RESULTS = OUTPUT / "fold0_results.json"
TRAINING_VERSION = "nested-selective-oof-definition-v26"
WEIGHTS = (0.0, 0.05, 0.10, 0.15, 0.20)


def _fold_f1(truth, score, count, folds):
    values = []
    for indices in folds:
        local_count = max(1, int(round(count * len(indices) / len(truth))))
        values.append(_label_f1(truth[indices], score[indices], local_count))
    return np.asarray(values)


def train_fold0():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    saved = np.load(FEATURES)
    if str(saved["training_version"]) != FEATURE_VERSION:
        raise RuntimeError("Run V25 to create matching leak-free OOF features")
    features = saved["features"].astype(np.float32)
    frame = load_train_data().reset_index(drop=True)
    targets = np.vstack(frame.factor_vector.to_numpy()).astype(np.int8)
    risk = frame.risk_label.to_numpy(dtype=np.int64)
    groups = frame.anon_user_id.astype(str).to_numpy()
    folds = list(StratifiedGroupKFold(
        n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED,
    ).split(np.zeros(len(frame)), risk, groups))
    outer_train, outer_valid = folds[0]
    current, _ = _current_v3_probability()

    inner_definition = _inner_oof(features, targets, risk, groups, outer_train)
    current_inner_rank = _percentile_rank(current[outer_train])
    definition_inner_rank = _percentile_rank(inner_definition)
    prevalence = targets[outer_train].mean(0)
    inner_groups = groups[outer_train]
    unique_groups = np.unique(inner_groups)
    rng = np.random.default_rng(config.SEED + 2600)
    rng.shuffle(unique_groups)
    group_blocks = np.array_split(unique_groups, 4)
    stability_folds = [
        np.flatnonzero(np.isin(inner_groups, block)) for block in group_blocks
    ]

    selected = np.zeros(config.NUM_FACTORS, dtype=np.float32)
    selection = []
    for label in range(config.NUM_FACTORS):
        truth = targets[outer_train, label]
        count = max(1, int(round(len(outer_train) * prevalence[label] * 1.10)))
        base_f1 = _label_f1(truth, current_inner_rank[:, label], count)
        base_auc = (roc_auc_score(truth, current_inner_rank[:, label])
                    if np.unique(truth).size == 2 else 0.5)
        definition_auc = (roc_auc_score(truth, definition_inner_rank[:, label])
                           if np.unique(truth).size == 2 else 0.5)
        base_fold = _fold_f1(
            truth, current_inner_rank[:, label], count, stability_folds
        )
        choices = []
        for weight in WEIGHTS:
            score = ((1.0 - weight) * current_inner_rank[:, label]
                     + weight * definition_inner_rank[:, label])
            total = _label_f1(truth, score, count)
            fold_values = _fold_f1(truth, score, count, stability_folds)
            choices.append((total, int((fold_values > base_fold).sum()), weight))
        best_total, positive_folds, best_weight = max(
            choices, key=lambda row: (row[0], row[1], -row[2])
        )
        support = int(truth.sum())
        required = 0.06 if support < 20 else 0.015
        if (best_total < base_f1 + required
                or definition_auc < base_auc + 0.003
                or positive_folds < 2):
            best_weight = 0.0
            best_total = base_f1
        selected[label] = best_weight
        selection.append({
            "label": config.ID2FACTOR[label], "support": support,
            "baseline_f1": float(base_f1), "selected_f1": float(best_total),
            "baseline_auc": float(base_auc), "definition_auc": float(definition_auc),
            "positive_stability_blocks": int(positive_folds),
            "selected_weight": float(best_weight),
        })

    outer_definition, _ = _fit_rankers(
        features, targets, outer_train, outer_valid
    )
    current_rank = _percentile_rank(current[outer_valid])
    definition_rank = _percentile_rank(outer_definition)
    candidate_score = ((1.0 - selected[None, :]) * current_rank
                       + selected[None, :] * definition_rank)
    baseline_prediction = _rank_decode(current[outer_valid], prevalence, 1.10)
    candidate_prediction = _rank_decode(candidate_score, prevalence, 1.10)
    baseline = float(f1_score(
        targets[outer_valid], baseline_prediction, average="macro", zero_division=0
    ))
    candidate = float(f1_score(
        targets[outer_valid], candidate_prediction, average="macro", zero_division=0
    ))
    per_label = [{
        "label": config.ID2FACTOR[label],
        "support": int(targets[outer_valid, label].sum()),
        "selected_weight": float(selected[label]),
        "baseline_f1": float(f1_score(
            targets[outer_valid, label], baseline_prediction[:, label], zero_division=0
        )),
        "candidate_f1": float(f1_score(
            targets[outer_valid, label], candidate_prediction[:, label], zero_division=0
        )),
    } for label in range(config.NUM_FACTORS)]
    payload = {
        "training_version": TRAINING_VERSION,
        "evaluation": "leak-free OOF features and nested label selection; outer fold0 untouched",
        "prevalence_ratio": 1.10,
        "selected_nonzero_labels": int((selected > 0).sum()),
        "baseline_macro_f1": baseline,
        "candidate_macro_f1": candidate,
        "delta": candidate - baseline,
        "promising_for_full_oof": bool(candidate >= baseline + .005),
        "adopted": False,
        "inner_selection": selection,
        "outer_per_label": per_label,
    }
    RESULTS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    return payload


if __name__ == "__main__":
    train_fold0()
