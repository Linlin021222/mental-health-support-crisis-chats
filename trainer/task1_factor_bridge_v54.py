"""Leak-conscious Task2-to-Task1 bridge evaluated on untouched users (V54).

The risk mapper is fitted only on gold factors belonging to the outer training
users.  At validation time it receives factor predictions produced by the
accepted factor fold that did not train on those users.  The fixed 5% blend is
intentionally conservative because the factor taxonomy does not encode all
four risk-level definitions.
"""
from __future__ import annotations

import json

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold

from analyze_task1_lexical_v11 import _lexical_experts, _transformer_probability
from baseline import _post_phrase_f1
from configs.config import config
from datasets.dataset import SuicideRiskDataset
from preprocess.preprocess import load_train_data
from trainer.factor_cross_encoder_v2 import _rank_decode
from trainer.task1_atomic_v25 import _load_records
from trainer.task1_qwen7b_verbalizer_v53 import _ensemble_prediction
from trainer.task1_risk_only_v27 import _v18_evidence
from utils.task1_metric import task1_score


OUTPUT = config.OUTPUT_DIR / "task1_factor_bridge_v54"
RESULTS = OUTPUT / "results.json"
FACTOR_WEIGHT = 0.05
V52_WEIGHT = 0.10


def _accepted_factor_oof():
    base_saved = np.load(config.OUTPUT_DIR / "factor_cv" / "factor_oof_predictions.npz")
    base = (config.FACTOR_SEMANTIC_MODEL_WEIGHT * base_saved["semantic"]
            + config.FACTOR_CPU_ENSEMBLE_WEIGHT * base_saved["cpu"])
    old = np.load(config.OUTPUT_DIR / "factor_cross_encoder" / "oof_predictions.npz")[
        "probabilities"
    ]
    new = np.load(config.OUTPUT_DIR / "factor_cross_encoder_v2" / "oof_predictions.npz")[
        "probabilities"
    ]
    calibration = json.loads((
        config.OUTPUT_DIR / "factor_cross_encoder_v2" / "calibration.json"
    ).read_text(encoding="utf-8"))
    probability = (float(calibration["base_weight"]) * base
                   + float(calibration["old_cross_weight"]) * old
                   + float(calibration["new_cross_weight"]) * new)
    return probability.astype(np.float32), calibration


def _select_c(factors, labels, groups, indices):
    """Choose regularization entirely inside the outer training users."""
    candidates = (0.01, 0.03, 0.10, 0.30, 1.0)
    rows = []
    local_y = labels[indices]
    local_g = groups[indices]
    folds = StratifiedGroupKFold(
        n_splits=4, shuffle=True, random_state=config.SEED + 5454,
    )
    for c in candidates:
        prediction = np.full(len(indices), -1, dtype=np.int64)
        for fit_pos, valid_pos in folds.split(
                np.zeros(len(indices)), local_y, groups=local_g):
            model = LogisticRegression(
                C=c, class_weight="balanced", max_iter=5000, random_state=config.SEED,
            )
            model.fit(factors[indices[fit_pos]], local_y[fit_pos])
            prediction[valid_pos] = model.predict(factors[indices[valid_pos]])
        rows.append({"c": c, "inner_weighted_f1": float(f1_score(
            local_y, prediction, average="weighted", zero_division=0,
        ))})
    selected = max(rows, key=lambda row: (row["inner_weighted_f1"], -row["c"]))
    return float(selected["c"]), rows


def _metric(truth, risk, phrase):
    risk_f1 = float(f1_score(truth, risk, average="weighted", zero_division=0))
    phrase_f1 = float(np.mean(phrase))
    return {"risk_f1": risk_f1, "phrase_f1": phrase_f1,
            "task1": task1_score(risk_f1, phrase_f1)}


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame = load_train_data().reset_index(drop=True)
    labels = frame.risk_label.to_numpy(dtype=np.int64)
    groups = frame.anon_user_id.astype(str).to_numpy()
    factors = np.vstack(frame.factor_vector.to_numpy()).astype(np.float32)
    train_idx, valid_idx = next(StratifiedGroupKFold(
        n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED,
    ).split(np.zeros(len(frame)), labels, groups))

    selected_c, inner_grid = _select_c(factors, labels, groups, train_idx)
    mapper = LogisticRegression(
        C=selected_c, class_weight="balanced", max_iter=5000,
        random_state=config.SEED,
    ).fit(factors[train_idx], labels[train_idx])

    factor_oof, factor_calibration = _accepted_factor_oof()
    factor_binary = _rank_decode(
        factor_oof[valid_idx], factors[train_idx].mean(0),
        float(factor_calibration["prevalence_ratio"]),
    ).astype(np.float32)
    factor_risk_probability = mapper.predict_proba(factor_binary)
    ordered = np.zeros((len(valid_idx), config.NUM_RISK_CLASSES), dtype=np.float64)
    ordered[:, mapper.classes_.astype(int)] = factor_risk_probability

    dataset = SuicideRiskDataset(config.CACHE_DIR / "train_cache.pt")
    _, _, outer_raw = _load_records()
    records = outer_raw["records"]
    transformer = _transformer_probability(dataset, valid_idx, records)
    v52_saved = torch.load(
        config.OUTPUT_DIR / "task1_rationale_augment_v52" / "strict_predictions.pt",
        map_location="cpu", weights_only=False,
    )
    if not np.array_equal(np.asarray(v52_saved["valid_idx"]), np.asarray(valid_idx)):
        raise RuntimeError("V52 and V54 outer folds do not match")
    v52_probability = np.vstack([row["probability"] for row in v52_saved["rows"]])
    v36 = json.loads((config.OUTPUT_DIR / "task1_oof_risk_v36" / "calibration.json")
                     .read_text(encoding="utf-8"))
    lexical = _lexical_experts(frame, train_idx, valid_idx)[v36["expert"]]
    texts = frame.text.iloc[valid_idx].astype(str).tolist()
    baseline = _ensemble_prediction(
        texts, transformer, v52_probability, ordered, lexical, v36, 0.0,
    )
    candidate = _ensemble_prediction(
        texts, transformer, v52_probability, ordered, lexical, v36, FACTOR_WEIGHT,
    )

    v18 = json.loads((config.OUTPUT_DIR / "task1_candidate_v18" / "results.json")
                     .read_text(encoding="utf-8"))
    v35 = json.loads((config.OUTPUT_DIR / "task1_oof_decoder_v35" / "calibration.json")
                     .read_text(encoding="utf-8"))
    evidence_parameters = (v35["parameters_by_predicted_risk"]
                           if v35.get("adopted", False)
                           else v18["evidence_parameters_by_predicted_risk"])
    seed2 = torch.load(
        config.OUTPUT_DIR / "task1_seed_ensemble_v14" / "seed2_valid.pt",
        map_location="cpu", weights_only=False,
    )["rows"]
    old_evidence = _v18_evidence(records, seed2, baseline, evidence_parameters)
    new_evidence = _v18_evidence(records, seed2, candidate, evidence_parameters)
    gold = [list(frame.iloc[int(index)].evidence) for index in valid_idx]
    old_phrase = np.asarray([_post_phrase_f1(x, y) for x, y in zip(old_evidence, gold)])
    new_phrase = np.asarray([_post_phrase_f1(x, y) for x, y in zip(new_evidence, gold)])
    truth = labels[valid_idx]
    base = _metric(truth, baseline, old_phrase)
    fixed = _metric(truth, candidate, new_phrase)

    unique_users = np.unique(groups[valid_idx])
    rng = np.random.default_rng(config.SEED + 5454)
    deltas = []
    for _ in range(4000):
        sampled = rng.choice(unique_users, size=len(unique_users), replace=True)
        positions = np.concatenate([
            np.flatnonzero(groups[valid_idx] == user) for user in sampled
        ])
        old_risk = f1_score(truth[positions], baseline[positions],
                            average="weighted", zero_division=0)
        new_risk = f1_score(truth[positions], candidate[positions],
                            average="weighted", zero_division=0)
        deltas.append(task1_score(new_risk, float(new_phrase[positions].mean()))
                      - task1_score(old_risk, float(old_phrase[positions].mean())))
    deltas = np.asarray(deltas)
    bootstrap = {"mean_delta": float(deltas.mean()),
                 "p05_delta": float(np.quantile(deltas, .05)),
                 "p95_delta": float(np.quantile(deltas, .95)),
                 "positive_fraction": float((deltas > 0).mean())}
    adopted = bool(fixed["task1"] >= base["task1"] + .003
                   and bootstrap["positive_fraction"] >= .80)
    payload = {
        "training_version": "task1-factor-risk-bridge-v54",
        "evaluation_scope": "one untouched outer user fold; current V52 baseline",
        "method": {"factor_model": "accepted prototype-MIL V3 OOF ensemble",
                   "risk_mapper": "balanced multinomial logistic regression",
                   "selected_c_inside_outer_train": selected_c,
                   "fixed_factor_weight": FACTOR_WEIGHT,
                   "v52_weight": V52_WEIGHT},
        "inner_regularization_grid": inner_grid,
        "factor_mapper_standalone_risk_f1": float(f1_score(
            truth, ordered.argmax(1), average="weighted", zero_division=0)),
        "baseline_v52": base,
        "fixed_candidate": {**fixed,
                            "changed_predictions": int(np.sum(baseline != candidate)),
                            "confusion": confusion_matrix(
                                truth, candidate, labels=np.arange(4)).tolist()},
        "user_cluster_bootstrap": bootstrap,
        "adopted": adopted,
    }
    RESULTS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    return payload


if __name__ == "__main__":
    main()
