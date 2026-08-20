"""Strict fold-0 TF-IDF probability fusion with the accepted Task 1 models."""
import json

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

from analyze_task1_v2_ensemble import collect
from baseline import _apply_task1_rules, _post_phrase_f1, _vectorizer
from configs.config import config
from datasets.dataset import SuicideRiskDataset
from models.multitask_model import SuicideRiskMultiTaskModel
from models.multitask_model_v2 import SuicideRiskMultiTaskModelV2
from preprocess.preprocess import load_train_data
from utils.task1_metric import task1_score as competition_task1_score


OUTPUT = config.OUTPUT_DIR / "task1_tfidf_hybrid"


def _softmax(values, temperature):
    values = values / float(temperature)
    values = values - values.max(1, keepdims=True)
    values = np.exp(values)
    return values / values.sum(1, keepdims=True)


def _score(old, v2, lexical_probability, weight, return_details=False):
    truth, prediction, phrases = [], [], []
    for index, (stable, ordinal) in enumerate(zip(old, v2)):
        v2_probability = 0.75 * ordinal["standard"] + 0.25 * ordinal["ordinal"]
        transformer = 0.8 * stable["standard"] + 0.2 * v2_probability
        probability = (
            (1.0 - weight) * transformer + weight * lexical_probability[index]
        )
        risk, evidence = _apply_task1_rules(
            stable["text"], int(np.argmax(probability)), stable["span"]
        )
        truth.append(stable["truth"]); prediction.append(risk)
        phrases.append(_post_phrase_f1(evidence, stable["gold"]))
    risk_f1 = float(f1_score(truth, prediction, average="weighted", zero_division=0))
    phrase_f1 = float(np.mean(phrases))
    metric = {"risk_f1": risk_f1, "phrase_f1": phrase_f1,
              "task1": competition_task1_score(risk_f1, phrase_f1)}
    if return_details:
        return metric, np.asarray(truth), np.asarray(prediction), np.asarray(phrases)
    return metric


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    dataset = SuicideRiskDataset(config.CACHE_DIR / "train_cache.pt")
    labels = np.asarray([int(x["risk_label"]) for x in dataset.data])
    groups = np.asarray([x["anon_user_id"] for x in dataset.data])
    train_idx, valid_idx = next(StratifiedGroupKFold(
        n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED
    ).split(np.zeros(len(dataset)), labels, groups))
    frame = load_train_data().reset_index(drop=True)

    vectorizer = _vectorizer()
    train_matrix = vectorizer.fit_transform(frame.text.iloc[train_idx])
    valid_matrix = vectorizer.transform(frame.text.iloc[valid_idx])
    logistic = OneVsRestClassifier(LogisticRegression(
        C=2.0, class_weight="balanced", max_iter=1000, solver="liblinear"
    )).fit(train_matrix, labels[train_idx])
    svc = LinearSVC(C=1.0, class_weight="balanced").fit(
        train_matrix, labels[train_idx]
    )
    lexical = {"logistic": logistic.predict_proba(valid_matrix)}
    decision = svc.decision_function(valid_matrix)
    for temperature in (0.5, 1.0, 1.5, 2.0):
        lexical[f"svc_temperature_{temperature}"] = _softmax(decision, temperature)

    device = torch.device(config.DEVICE)
    old_model = SuicideRiskMultiTaskModel().to(device)
    old_model.load_state_dict(torch.load(config.OUTPUT_DIR / "best_model.pt", map_location=device))
    old = collect(old_model, dataset, valid_idx, device)
    del old_model
    if device.type == "cuda": torch.cuda.empty_cache()
    v2_model = SuicideRiskMultiTaskModelV2().to(device)
    v2_model.load_state_dict(torch.load(
        config.OUTPUT_DIR / "task1_v2_strict_model.pt", map_location=device
    ))
    v2 = collect(v2_model, dataset, valid_idx, device, v2=True)
    del v2_model
    if device.type == "cuda": torch.cuda.empty_cache()

    rows = []
    for name, probabilities in lexical.items():
        lexical_only = f1_score(
            labels[valid_idx], probabilities.argmax(1), average="weighted", zero_division=0
        )
        for weight in (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30):
            rows.append({"lexical_model": name, "lexical_only_risk_f1": float(lexical_only),
                         "lexical_weight": weight, **_score(old, v2, probabilities, weight)})
    rows.sort(key=lambda item: item["task1"], reverse=True)
    baseline = next(item for item in rows if item["lexical_weight"] == 0.0)
    best = rows[0]
    _, truth, baseline_prediction, baseline_phrase = _score(
        old, v2, lexical["logistic"], 0.0, return_details=True
    )
    _, _, best_prediction, best_phrase = _score(
        old, v2, lexical[best["lexical_model"]], best["lexical_weight"],
        return_details=True,
    )
    # Cluster bootstrap by anon_user_id so repeated posts from one author are
    # never treated as independent evidence of stability.
    valid_groups = groups[valid_idx].astype(str)
    unique_groups = np.unique(valid_groups)
    rng = np.random.default_rng(config.SEED + 901)
    deltas = []
    for _ in range(1000):
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        sampled = np.concatenate([
            np.flatnonzero(valid_groups == group) for group in sampled_groups
        ])
        old_risk = f1_score(
            truth[sampled], baseline_prediction[sampled], average="weighted", zero_division=0
        )
        new_risk = f1_score(
            truth[sampled], best_prediction[sampled], average="weighted", zero_division=0
        )
        old_task1 = competition_task1_score(old_risk, baseline_phrase[sampled].mean())
        new_task1 = competition_task1_score(new_risk, best_phrase[sampled].mean())
        deltas.append(float(new_task1 - old_task1))
    bootstrap = {
        "mean_delta": float(np.mean(deltas)),
        "p05_delta": float(np.quantile(deltas, 0.05)),
        "p95_delta": float(np.quantile(deltas, 0.95)),
        "positive_fraction": float(np.mean(np.asarray(deltas) > 0)),
        "user_count": int(len(unique_groups)),
    }
    adopted = bool(
        best["lexical_weight"] > 0
        and best["task1"] >= baseline["task1"] + 0.003
        and bootstrap["positive_fraction"] >= 0.75
    )
    payload = {
        "baseline": baseline, "best_optimistic": best,
        "optimistic_delta": best["task1"] - baseline["task1"],
        "user_cluster_bootstrap": bootstrap, "adopted": adopted,
        "top15": rows[:15],
    }
    (OUTPUT / "fold0_results.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    (OUTPUT / "calibration.json").write_text(json.dumps({
        "adopted": adopted, "training_version": "tfidf-risk-hybrid-v1",
        "lexical_model": best["lexical_model"],
        "lexical_weight": best["lexical_weight"],
        "temperature": 0.5 if best["lexical_model"] == "svc_temperature_0.5" else None,
        "strict_task1": best["task1"], "strict_baseline_task1": baseline["task1"],
        "strict_delta": best["task1"] - baseline["task1"],
        "bootstrap_positive_fraction": bootstrap["positive_fraction"],
    }, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2)); return payload


if __name__ == "__main__":
    main()
