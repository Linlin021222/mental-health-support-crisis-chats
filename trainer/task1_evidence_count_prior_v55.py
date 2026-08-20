"""Training-only evidence-count prior for metric-aligned decoding (V55)."""
from __future__ import annotations

import json

import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

from analyze_task1_lexical_v11 import _lexical_experts, _transformer_probability
from baseline import _post_phrase_f1
from configs.config import config
from datasets.dataset import SuicideRiskDataset
from preprocess.preprocess import load_train_data
from trainer.task1_atomic_v25 import _load_records
from trainer.task1_qwen7b_verbalizer_v53 import _ensemble_prediction
from trainer.task1_risk_only_v27 import _v18_evidence
from utils.task1_metric import task1_score


OUTPUT = config.OUTPUT_DIR / "task1_evidence_count_prior_v55"
RESULTS = OUTPUT / "results.json"


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
    train_idx, valid_idx = next(StratifiedGroupKFold(
        n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED,
    ).split(np.zeros(len(frame)), labels, groups))

    # Fixed, training-only 75th-percentile prior. Indicator remains empty to
    # avoid sacrificing the overwhelming majority of empty gold evidence.
    caps = {0: 0}
    count_summary = {}
    for risk in range(4):
        counts = np.asarray([
            len(frame.iloc[int(index)].evidence)
            for index in train_idx if labels[int(index)] == risk
        ])
        count_summary[config.ID2RISK[risk]] = {
            "posts": int(len(counts)), "mean": float(counts.mean()),
            "median": float(np.median(counts)),
            "p75": float(np.quantile(counts, .75)),
        }
        if risk:
            caps[risk] = max(1, int(np.ceil(np.quantile(counts, .75))))

    dataset = SuicideRiskDataset(config.CACHE_DIR / "train_cache.pt")
    _, _, outer_raw = _load_records()
    records = outer_raw["records"]
    transformer = _transformer_probability(dataset, valid_idx, records)
    v52_saved = torch.load(
        config.OUTPUT_DIR / "task1_rationale_augment_v52" / "strict_predictions.pt",
        map_location="cpu", weights_only=False,
    )
    v52_probability = np.vstack([row["probability"] for row in v52_saved["rows"]])
    v36 = json.loads((config.OUTPUT_DIR / "task1_oof_risk_v36" / "calibration.json")
                     .read_text(encoding="utf-8"))
    lexical = _lexical_experts(frame, train_idx, valid_idx)[v36["expert"]]
    texts = frame.text.iloc[valid_idx].astype(str).tolist()
    dummy = np.full_like(transformer, 1.0 / config.NUM_RISK_CLASSES)
    risks = _ensemble_prediction(
        texts, transformer, v52_probability, dummy, lexical, v36, 0.0,
    )

    v18 = json.loads((config.OUTPUT_DIR / "task1_candidate_v18" / "results.json")
                     .read_text(encoding="utf-8"))
    v35 = json.loads((config.OUTPUT_DIR / "task1_oof_decoder_v35" / "calibration.json")
                     .read_text(encoding="utf-8"))
    parameters = (v35["parameters_by_predicted_risk"]
                  if v35.get("adopted", False)
                  else v18["evidence_parameters_by_predicted_risk"])
    seed2 = torch.load(
        config.OUTPUT_DIR / "task1_seed_ensemble_v14" / "seed2_valid.pt",
        map_location="cpu", weights_only=False,
    )["rows"]
    baseline_evidence = _v18_evidence(records, seed2, risks, parameters)
    candidate_evidence = [
        phrases[:caps[int(risk)]] for phrases, risk in zip(baseline_evidence, risks)
    ]
    gold = [list(frame.iloc[int(index)].evidence) for index in valid_idx]
    old_phrase = np.asarray([
        _post_phrase_f1(predicted, target)
        for predicted, target in zip(baseline_evidence, gold)
    ])
    new_phrase = np.asarray([
        _post_phrase_f1(predicted, target)
        for predicted, target in zip(candidate_evidence, gold)
    ])
    truth = labels[valid_idx]
    base = _metric(truth, risks, old_phrase)
    fixed = _metric(truth, risks, new_phrase)

    unique_users = np.unique(groups[valid_idx])
    rng = np.random.default_rng(config.SEED + 5555)
    deltas = []
    for _ in range(4000):
        sampled = rng.choice(unique_users, size=len(unique_users), replace=True)
        positions = np.concatenate([
            np.flatnonzero(groups[valid_idx] == user) for user in sampled
        ])
        deltas.append(task1_score(
            f1_score(truth[positions], risks[positions], average="weighted",
                     zero_division=0), float(new_phrase[positions].mean()),
        ) - task1_score(
            f1_score(truth[positions], risks[positions], average="weighted",
                     zero_division=0), float(old_phrase[positions].mean()),
        ))
    deltas = np.asarray(deltas)
    bootstrap = {"mean_delta": float(deltas.mean()),
                 "p05_delta": float(np.quantile(deltas, .05)),
                 "p95_delta": float(np.quantile(deltas, .95)),
                 "positive_fraction": float((deltas > 0).mean())}
    adopted = bool(fixed["task1"] >= base["task1"] + .003
                   and bootstrap["positive_fraction"] >= .80)
    payload = {
        "training_version": "task1-training-count-prior-v55",
        "evaluation_scope": "one untouched outer user fold; current V52/V35 baseline",
        "method": {"count_statistic": "training-only p75", "caps": {
            config.ID2RISK[key]: value for key, value in caps.items()}},
        "training_count_summary": count_summary,
        "baseline": base,
        "candidate": {**fixed,
                      "changed_posts": int(sum(a != b for a, b in zip(
                          baseline_evidence, candidate_evidence))),
                      "improved_posts": int(np.sum(new_phrase > old_phrase)),
                      "worsened_posts": int(np.sum(new_phrase < old_phrase))},
        "user_cluster_bootstrap": bootstrap,
        "adopted": adopted,
    }
    RESULTS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    return payload


if __name__ == "__main__":
    main()
