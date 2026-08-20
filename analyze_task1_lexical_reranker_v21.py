"""Lexical candidate reranker trained from V20's complete nested OOF data."""
from __future__ import annotations

import json

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import FeatureUnion

from baseline import _post_phrase_f1
from configs.config import config
from datasets.dataset import SuicideRiskDataset
from inference.task1_evidence_v4 import load_evidence_calibration
from trainer.task1_evidence_reranker_v13 import (
    _pair_rows,
    _pool_records,
    _post_score_map,
)
from trainer.task1_oof_stack_v20 import (
    INNER_FOLDS,
    OUTPUT as V20_OUTPUT,
    _baseline_evidence,
    _hybrid_prediction,
    _parameter_grid,
)
from utils.task1_metric import task1_score


OUTPUT = config.OUTPUT_DIR / "task1_lexical_reranker_v21"
RESULTS = OUTPUT / "results.json"
CALIBRATION = OUTPUT / "calibration.json"
MODEL = OUTPUT / "model.pt"
TRAINING_VERSION = "task1-nested-oof-lexical-v21"


def _load_inner_records():
    records = []
    membership = {}
    for fold in range(INNER_FOLDS):
        saved = torch.load(
            V20_OUTPUT / f"inner_fold{fold}_raw.pt",
            map_location="cpu", weights_only=False,
        )
        for row in saved["records"]:
            records.append(row)
            membership[int(row["global_index"])] = fold
    records.sort(key=lambda row: row["global_index"])
    folds = np.asarray([
        membership[int(row["global_index"])] for row in records
    ], dtype=np.int64)
    return records, folds


def _outer_records(dataset, outer_valid):
    raw = torch.load(
        config.OUTPUT_DIR / "task1_evidence_v4" / "fold0_raw.pt",
        map_location="cpu", weights_only=False,
    )
    if not np.array_equal(np.asarray(raw["valid_idx"]), outer_valid):
        raise ValueError("V21 outer rows differ from the stable holdout")
    return raw["records"]


def _documents(rows):
    # Candidate comes first so its exact lexical form is not diluted by a long
    # Reddit context.  The structured prompt also contributes predicted-risk
    # and first-stage agreement buckets.
    return [f"{row['prompt']} LOCALCONTEXT {row['context']}" for row in rows]


def _vectorizer():
    return FeatureUnion([
        ("char", TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 5), min_df=2,
            max_features=180_000, sublinear_tf=True, dtype=np.float32,
        )),
        ("word", TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), min_df=2,
            max_features=90_000, sublinear_tf=True, strip_accents="unicode",
            dtype=np.float32,
        )),
    ])


def _fit(rows, c_value):
    vectorizer = _vectorizer()
    matrix = vectorizer.fit_transform(_documents(rows))
    labels = np.asarray([row["label"] for row in rows], dtype=np.int64)
    model = LogisticRegression(
        C=float(c_value), class_weight="balanced", max_iter=1000,
        solver="liblinear", random_state=config.SEED + 2121,
    ).fit(matrix, labels)
    return vectorizer, model


def _probability(vectorizer, model, rows):
    return model.predict_proba(vectorizer.transform(_documents(rows)))[:, 1]


def _predict(posts, rows, scores, parameters):
    grouped = _post_score_map(rows, scores)
    baseline_scores = []; candidate_scores = []; evidence = []
    for index, post in enumerate(posts):
        baseline = list(post["baseline_evidence"])
        values = [
            grouped[index].get(candidate_index, 0.0)
            for candidate_index in range(len(post["candidates"]))
        ]
        selected = _hybrid_prediction(post, values, baseline, parameters)
        evidence.append(selected)
        baseline_scores.append(_post_phrase_f1(baseline, post["gold"]))
        candidate_scores.append(_post_phrase_f1(selected, post["gold"]))
    return (
        evidence,
        np.asarray(baseline_scores, dtype=np.float32),
        np.asarray(candidate_scores, dtype=np.float32),
    )


def _bootstrap(groups, baseline, candidate):
    unique = np.unique(groups)
    rng = np.random.default_rng(config.SEED + 2121)
    values = []
    for _ in range(4000):
        sampled_users = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([
            np.flatnonzero(groups == user) for user in sampled_users
        ])
        values.append(float(candidate[indices].mean() - baseline[indices].mean()))
    values = np.asarray(values)
    return {
        "mean_phrase_delta": float(values.mean()),
        "p05_phrase_delta": float(np.quantile(values, 0.05)),
        "p95_phrase_delta": float(np.quantile(values, 0.95)),
        "positive_fraction": float((values > 0).mean()),
    }


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    dataset = SuicideRiskDataset(config.CACHE_DIR / "train_cache.pt")
    labels = np.asarray([int(row["risk_label"]) for row in dataset.data])
    groups = np.asarray([str(row["anon_user_id"]) for row in dataset.data])
    outer_train, outer_valid = next(StratifiedGroupKFold(
        n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED
    ).split(np.zeros(len(dataset)), labels, groups))
    evidence_calibration = load_evidence_calibration()
    inner_records, fold_membership = _load_inner_records()
    outer_records = _outer_records(dataset, outer_valid)
    inner_posts = _pool_records(inner_records, use_truth=False)
    outer_posts = _pool_records(outer_records, use_truth=False)
    for post, record in zip(inner_posts, inner_records):
        post["baseline_evidence"] = _baseline_evidence(record, evidence_calibration)
    for post, record in zip(outer_posts, outer_records):
        post["baseline_evidence"] = _baseline_evidence(record, evidence_calibration)

    calibration_fold = INNER_FOLDS - 1
    fit_posts = np.flatnonzero(fold_membership != calibration_fold)
    calibration_posts = np.flatnonzero(fold_membership == calibration_fold)
    fit_rows = _pair_rows(inner_posts, fit_posts, training=True)
    calibration_rows = _pair_rows(
        inner_posts, calibration_posts, training=False
    )
    calibration_history = []
    for c_value in (0.25, 0.50, 1.0, 2.0, 4.0):
        vectorizer, model = _fit(fit_rows, c_value)
        probabilities = _probability(
            vectorizer, model, calibration_rows
        )
        selected = _parameter_grid(
            inner_posts, calibration_rows, probabilities,
            calibration_posts, evidence_calibration,
        )[0]
        row = {"C": c_value, **selected}
        calibration_history.append(row)
        print(
            f"V21 C={c_value:.2f} inner_phrase={row['phrase_f1']:.4f} "
            f"mode={row['mode']} topk={row['topk']} "
            f"threshold={row['threshold']:.2f} gate={row['gate']:.2f}",
            flush=True,
        )
    chosen = max(calibration_history, key=lambda row: row["phrase_f1"])

    # Outer holdout remains untouched, so after selecting C/policy on the
    # fourth inner fold we can refit the lexical scorer on all nested OOF
    # candidate pairs before its single outer evaluation.
    all_rows = _pair_rows(
        inner_posts, np.arange(len(inner_posts)), training=True
    )
    vectorizer, model = _fit(all_rows, chosen["C"])
    outer_rows = _pair_rows(
        outer_posts, np.arange(len(outer_posts)), training=False
    )
    outer_probability = _probability(vectorizer, model, outer_rows)
    predictions, baseline_phrase, candidate_phrase = _predict(
        outer_posts, outer_rows, outer_probability, chosen
    )
    truth = labels[outer_valid]
    risk = np.asarray([int(row["risk"]) for row in outer_records])
    risk_f1 = float(f1_score(
        truth, risk, average="weighted", zero_division=0
    ))
    baseline_task1 = task1_score(risk_f1, float(baseline_phrase.mean()))
    candidate_task1 = task1_score(risk_f1, float(candidate_phrase.mean()))
    bootstrap = _bootstrap(
        groups[outer_valid], baseline_phrase, candidate_phrase
    )
    adopted = bool(
        candidate_task1 >= baseline_task1 + 0.005
        and bootstrap["positive_fraction"] >= 0.80
    )
    payload = {
        "training_version": TRAINING_VERSION,
        "evaluation_scope": "complete nested OOF lexical fit; untouched outer users",
        "fit_pairs": len(all_rows),
        "calibration_history": calibration_history,
        "selected": chosen,
        "baseline": {
            "risk_f1": risk_f1,
            "phrase_f1": float(baseline_phrase.mean()),
            "task1": baseline_task1,
        },
        "candidate": {
            "risk_f1": risk_f1,
            "phrase_f1": float(candidate_phrase.mean()),
            "task1": candidate_task1,
            "improved_posts": int((candidate_phrase > baseline_phrase).sum()),
            "worsened_posts": int((candidate_phrase < baseline_phrase).sum()),
        },
        "user_cluster_bootstrap": bootstrap,
        "adopted": adopted,
    }
    torch.save({
        "training_version": TRAINING_VERSION,
        "vectorizer": vectorizer, "model": model, "parameters": chosen,
    }, MODEL)
    RESULTS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    CALIBRATION.write_text(json.dumps({
        "training_version": TRAINING_VERSION, "adopted": adopted,
        "strict_task1": candidate_task1,
        "bootstrap_positive_fraction": bootstrap["positive_fraction"],
        **{key: chosen[key] for key in ("C", "mode", "topk", "threshold", "gate")},
    }, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    return payload


if __name__ == "__main__":
    main()
