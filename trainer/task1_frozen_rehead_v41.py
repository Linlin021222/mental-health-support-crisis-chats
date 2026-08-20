"""User-disjoint frozen-DeBERTa risk re-head experiment (V41)."""
from __future__ import annotations

from collections import Counter
import json

import joblib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from analyze_task1_oof_risk_v36 import CACHE as V36_CACHE, _evidence_matrix, _predict
from configs.config import config
from datasets.dataset import SuicideRiskDataset
from inference.task1_evidence_v4 import correct_risk_only
from models.multitask_model import SuicideRiskMultiTaskModel
from preprocess.preprocess import load_train_data
from trainer.task1_atomic_v25 import _load_records
from trainer.task1_boundary_model_v39 import _loader
from utils.task1_metric import task1_score


OUTPUT = config.OUTPUT_DIR / "task1_frozen_rehead_v41"
RESULTS = OUTPUT / "results.json"
CALIBRATION = OUTPUT / "calibration.json"
TRAINING_VERSION = "task1-frozen-deberta-regularized-rehead-v41"
V20 = config.OUTPUT_DIR / "task1_oof_stack_v20"


@torch.no_grad()
def _extract_fold(fold, dataset, device):
    cache = OUTPUT / f"fold{fold}_embeddings.npz"
    raw = torch.load(V20 / f"inner_fold{fold}_raw.pt", map_location="cpu",
                     weights_only=False)
    fit = np.asarray(raw["fit_idx"], dtype=np.int64)
    held = np.asarray(raw["oof_idx"], dtype=np.int64)
    if cache.exists():
        saved = np.load(cache)
        if (str(saved["training_version"].item()) == TRAINING_VERSION
                and np.array_equal(saved["fit_indices"], fit)
                and np.array_equal(saved["held_indices"], held)):
            print(f"V41 fold {fold}: resumed frozen embeddings", flush=True)
            return fit, held, saved["fit_embeddings"], saved["held_embeddings"]
    model = SuicideRiskMultiTaskModel()
    model.load_state_dict(torch.load(V20 / f"inner_fold{fold}_model.pt",
                                     map_location="cpu", weights_only=True))
    model.backbone.encoder.gradient_checkpointing_disable()
    model.to(device).eval()

    def collect(indices, description):
        rows = []
        for batch in tqdm(_loader(dataset, indices, False), desc=description):
            with torch.autocast(device_type="cuda", enabled=True):
                hidden = model.backbone(
                    batch["input_ids"].to(device), batch["attention_mask"].to(device)
                ).float()
                document = model.pooling(hidden, batch["attention_mask"].to(device))
            rows.append(document.float().cpu().numpy())
        return np.vstack(rows)

    fit_embeddings = collect(fit, f"V41 fold {fold} fit embeddings")
    held_embeddings = collect(held, f"V41 fold {fold} OOF embeddings")
    np.savez_compressed(cache, training_version=TRAINING_VERSION,
                        fit_indices=fit, held_indices=held,
                        fit_embeddings=fit_embeddings,
                        held_embeddings=held_embeddings)
    del model; torch.cuda.empty_cache()
    return fit, held, fit_embeddings, held_embeddings


def _head(c_value, balanced):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=float(c_value), class_weight="balanced" if balanced else None,
                           max_iter=2500, solver="lbfgs"),
    )


def _metric(truth, prediction, evidence, indices):
    indices = np.asarray(indices, dtype=np.int64)
    risk = float(f1_score(truth[indices], prediction[indices], average="weighted",
                          zero_division=0))
    phrase_values = evidence[np.arange(len(prediction)), prediction]
    phrase = float(phrase_values[indices].mean())
    return risk, phrase, task1_score(risk, phrase), phrase_values


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("V41 frozen risk re-head requires CUDA")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame = load_train_data().reset_index(drop=True)
    labels = frame.risk_label.to_numpy(dtype=np.int64)
    groups = frame.anon_user_id.astype(str).to_numpy()
    dataset = SuicideRiskDataset(config.CACHE_DIR / "train_cache.pt")
    records, membership_map, _ = _load_records()
    global_indices = np.asarray([int(row["global_index"]) for row in records])
    position = {int(index): pos for pos, index in enumerate(global_indices)}
    membership = np.asarray([membership_map[int(index)] for index in global_indices])
    truth = labels[global_indices]
    device = torch.device("cuda")

    specifications = [(c, balanced) for c in (.0003, .001, .003, .01, .03, .1)
                      for balanced in (False, True)]
    head_probability = {spec: np.zeros((len(records), 4), dtype=np.float32)
                        for spec in specifications}
    fold_models = {}
    for fold in range(4):
        fit, held, fit_x, held_x = _extract_fold(fold, dataset, device)
        positions = [position[int(index)] for index in held]
        for spec in specifications:
            model = _head(*spec).fit(fit_x, labels[fit])
            head_probability[spec][positions] = model.predict_proba(held_x)
            fold_models[(fold, spec)] = model
        print(f"V41 fold {fold}: fitted {len(specifications)} frozen heads", flush=True)

    saved = np.load(V36_CACHE, allow_pickle=True)
    names = saved["names"].tolist(); decisions = saved["decisions"]
    parameters = json.loads((config.OUTPUT_DIR / "task1_oof_risk_v36" / "calibration.json")
                            .read_text(encoding="utf-8"))
    lexical = decisions[names.index(parameters["expert"])]
    old_probability = np.vstack([row["old_probability"] for row in records])
    corrections = np.asarray([[correct_risk_only(row["text"], risk) for risk in range(4)]
                              for row in records], dtype=np.int64)
    evidence = _evidence_matrix(records)
    baseline = _predict(old_probability, lexical, parameters, corrections)
    base = _metric(truth, baseline, evidence, np.arange(len(truth)))

    weights = (.10, .25, .50, .75, 1.)
    crossfit = np.zeros(len(truth), dtype=np.int64); selections = []; folds = []
    for fold in range(4):
        fit_positions = np.flatnonzero(membership != fold)
        held_positions = np.flatnonzero(membership == fold)
        choices = []
        for spec in specifications:
            for weight in weights:
                blended = ((1. - weight) * old_probability
                           + weight * head_probability[spec])
                prediction = _predict(blended, lexical, parameters, corrections)
                score = _metric(truth, prediction, evidence, fit_positions)
                choices.append((score[2], score[0], spec, weight, prediction))
        _, _, spec, weight, prediction = max(choices, key=lambda row: (row[0], row[1]))
        crossfit[held_positions] = prediction[held_positions]
        old = _metric(truth, baseline, evidence, held_positions)
        new = _metric(truth, crossfit, evidence, held_positions)
        selections.append((spec, weight))
        folds.append({"fold": fold, "posts": int(len(held_positions)),
                      "c": spec[0], "balanced": spec[1], "new_head_weight": weight,
                      "baseline_risk_f1": old[0], "candidate_risk_f1": new[0],
                      "baseline_task1": old[2], "candidate_task1": new[2]})
        print(f"V41 fold={fold} C={spec[0]} balanced={spec[1]} weight={weight} "
              f"task1 {old[2]:.6f}->{new[2]:.6f}", flush=True)

    cross = _metric(truth, crossfit, evidence, np.arange(len(truth)))
    production_spec = Counter(spec for spec, _ in selections).most_common(1)[0][0]
    selected_weights = [weight for _, weight in selections]
    centre = float(np.median(selected_weights))
    production_weight = min(weights, key=lambda value: abs(value - centre))
    blended = ((1. - production_weight) * old_probability
               + production_weight * head_probability[production_spec])
    fixed_prediction = _predict(blended, lexical, parameters, corrections)
    fixed = _metric(truth, fixed_prediction, evidence, np.arange(len(truth)))

    local_groups = groups[global_indices]; unique = np.unique(local_groups)
    rng = np.random.default_rng(config.SEED + 4141); deltas = []
    for _ in range(4000):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        positions = np.concatenate([np.flatnonzero(local_groups == user) for user in sampled])
        old_risk = f1_score(truth[positions], baseline[positions], average="weighted",
                            zero_division=0)
        new_risk = f1_score(truth[positions], crossfit[positions], average="weighted",
                            zero_division=0)
        deltas.append(task1_score(new_risk, float(cross[3][positions].mean()))
                      - task1_score(old_risk, float(base[3][positions].mean())))
    deltas = np.asarray(deltas)
    bootstrap = {"mean_task1_delta": float(deltas.mean()),
                 "p05_task1_delta": float(np.quantile(deltas, .05)),
                 "p95_task1_delta": float(np.quantile(deltas, .95)),
                 "positive_fraction": float((deltas > 0).mean())}
    adopted = bool(cross[2] >= base[2] + .003 and fixed[2] >= base[2] + .002
                   and bootstrap["positive_fraction"] >= .80)
    payload = {"training_version": TRAINING_VERSION,
               "evaluation_scope": "all 1305 user-disjoint nested OOF posts",
               "baseline_v36": {"risk_f1": base[0], "phrase_f1": base[1],
                                "task1": base[2]},
               "crossfit_candidate": {"risk_f1": cross[0], "phrase_f1": cross[1],
                                      "task1": cross[2], "folds": folds},
               "fixed_production_diagnostic": {
                   "c": production_spec[0], "balanced": production_spec[1],
                   "new_head_weight": production_weight, "risk_f1": fixed[0],
                   "phrase_f1": fixed[1], "task1": fixed[2]},
               "user_cluster_bootstrap": bootstrap, "adopted": adopted}
    RESULTS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    CALIBRATION.write_text(json.dumps({"training_version": TRAINING_VERSION,
        "adopted": adopted, "c": production_spec[0], "balanced": production_spec[1],
        "new_head_weight": production_weight, "crossfit_task1": cross[2],
        "baseline_task1": base[2]}, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    return payload


if __name__ == "__main__":
    main()
