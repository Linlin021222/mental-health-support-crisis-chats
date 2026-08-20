"""Check whether the rejected Task1 rationale-v3 fold is ensemble-complementary."""
import json

import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

from analyze_task1_v2_ensemble import collect, deduplicate
from baseline import _apply_task1_rules, _post_phrase_f1
from configs.config import config
from datasets.dataset import SuicideRiskDataset
from models.multitask_model import SuicideRiskMultiTaskModel
from models.multitask_model_v2 import SuicideRiskMultiTaskModelV2
from utils.task1_metric import task1_score as competition_task1_score


def main():
    dataset = SuicideRiskDataset(config.CACHE_DIR / "train_cache.pt")
    labels = np.asarray([int(x["risk_label"]) for x in dataset.data])
    groups = np.asarray([x["anon_user_id"] for x in dataset.data])
    _, valid = next(StratifiedGroupKFold(
        n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED
    ).split(np.zeros(len(dataset)), labels, groups))
    saved = np.load(config.OUTPUT_DIR / "task1_cv" / "fold0_valid.npz", allow_pickle=True)
    if not np.array_equal(saved["valid_indices"], valid):
        raise ValueError("Task 1 rationale fold indices do not match the stable reference fold")

    device = torch.device(config.DEVICE)
    old_model = SuicideRiskMultiTaskModel().to(device)
    old_model.load_state_dict(torch.load(config.OUTPUT_DIR / "best_model.pt", map_location=device))
    old = collect(old_model, dataset, valid, device)
    del old_model
    if device.type == "cuda": torch.cuda.empty_cache()
    v2_model = SuicideRiskMultiTaskModelV2().to(device)
    v2_model.load_state_dict(torch.load(
        config.OUTPUT_DIR / "task1_v2_strict_model.pt", map_location=device
    ))
    v2 = collect(v2_model, dataset, valid, device, v2=True)
    del v2_model
    if device.type == "cuda": torch.cuda.empty_cache()

    new_probability = 0.75 * saved["standard"] + 0.25 * saved["ordinal"]
    new_evidence = saved["candidate"].tolist()
    results = []
    for new_weight in (0.0, 0.05, 0.10, 0.20, 0.30):
        for evidence_mode in ("old", "old_new", "new_old", "new"):
            truth, prediction, phrase_scores = [], [], []
            for i, (stable, ordinal) in enumerate(zip(old, v2)):
                v2_probability = 0.75 * ordinal["standard"] + 0.25 * ordinal["ordinal"]
                stable_probability = 0.8 * stable["standard"] + 0.2 * v2_probability
                probability = (
                    (1.0 - new_weight) * stable_probability
                    + new_weight * new_probability[i]
                )
                if evidence_mode == "old": evidence = stable["span"]
                elif evidence_mode == "new": evidence = list(new_evidence[i])[:3]
                elif evidence_mode == "old_new": evidence = deduplicate(
                    stable["span"] + list(new_evidence[i])
                )[:3]
                else: evidence = deduplicate(
                    list(new_evidence[i]) + stable["span"]
                )[:3]
                risk, evidence = _apply_task1_rules(
                    stable["text"], int(np.argmax(probability)), evidence
                )
                truth.append(stable["truth"]); prediction.append(risk)
                phrase_scores.append(_post_phrase_f1(evidence, stable["gold"]))
            risk_f1 = f1_score(truth, prediction, average="weighted", zero_division=0)
            phrase_f1 = float(np.mean(phrase_scores))
            results.append({
                "new_weight": new_weight, "evidence_mode": evidence_mode,
                "risk_f1": float(risk_f1), "phrase_f1": phrase_f1,
                "task1": competition_task1_score(risk_f1, phrase_f1),
            })
    results.sort(key=lambda item: item["task1"], reverse=True)
    baseline = next(
        item for item in results
        if item["new_weight"] == 0.0 and item["evidence_mode"] == "old"
    )
    payload = {
        "baseline": baseline, "best_optimistic": results[0],
        "optimistic_delta": results[0]["task1"] - baseline["task1"],
        "top10": results[:10],
    }
    output = config.OUTPUT_DIR / "task1_cv" / "fold0_ensemble_ablation.json"
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2)); return payload


if __name__ == "__main__":
    main()
