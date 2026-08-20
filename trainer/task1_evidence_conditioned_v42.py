"""Evidence-conditioned frozen risk model with nested user OOF gating (V42)."""
from __future__ import annotations

from collections import Counter
import json

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


OUTPUT = config.OUTPUT_DIR / "task1_evidence_conditioned_v42"
RESULTS = OUTPUT / "results.json"
CALIBRATION = OUTPUT / "calibration.json"
TRAINING_VERSION = "task1-evidence-conditioned-frozen-risk-v42"
V20 = config.OUTPUT_DIR / "task1_oof_stack_v20"


def _evidence_features(hidden, attention_mask, start_logits, end_logits, document):
    """Pool token states around evidence endpoints without using gold spans."""
    batch, chunks, length, width = hidden.shape
    valid = attention_mask.bool()
    endpoint_logits = torch.maximum(start_logits.float(), end_logits.float())
    endpoint_logits = endpoint_logits.masked_fill(~valid, -20.)
    flat_logits = endpoint_logits.reshape(batch, -1)
    flat_hidden = hidden.reshape(batch, chunks * length, width).float()
    flat_valid = valid.reshape(batch, -1)

    # Soft endpoint attention retains uncertainty; top-k pooling protects a
    # second evidence phrase from being hidden by one dominant endpoint.
    attention = torch.softmax(flat_logits / .35, dim=1)
    soft_evidence = (flat_hidden * attention.unsqueeze(-1)).sum(1)
    k = min(12, flat_logits.shape[1])
    top_values, top_indices = flat_logits.topk(k, dim=1)
    gather = top_indices.unsqueeze(-1).expand(-1, -1, width)
    top_hidden = flat_hidden.gather(1, gather)
    top_weight = torch.softmax(top_values / .5, dim=1)
    top_evidence = (top_hidden * top_weight.unsqueeze(-1)).sum(1)

    probability = torch.sigmoid(flat_logits).masked_fill(~flat_valid, 0.)
    count = flat_valid.sum(1).clamp_min(1)
    top5 = probability.topk(min(5, probability.shape[1]), dim=1).values.mean(1)
    peak_index = probability.argmax(1).float() / count.float()
    statistics = torch.stack((
        probability.max(1).values, top5,
        (probability >= .50).sum(1).float() / count,
        (probability >= .70).sum(1).float() / count,
        peak_index.clamp(0., 1.),
    ), 1)
    # The residual relation encodes whether the evidence area agrees with the
    # overall post semantics while keeping the feature map linear and small.
    return torch.cat((document.float(), soft_evidence, top_evidence - document.float(),
                      statistics), 1)


@torch.no_grad()
def _extract_fold(fold, dataset, device):
    cache = OUTPUT / f"fold{fold}_features.npz"
    raw = torch.load(V20 / f"inner_fold{fold}_raw.pt", map_location="cpu",
                     weights_only=False)
    fit = np.asarray(raw["fit_idx"], dtype=np.int64)
    held = np.asarray(raw["oof_idx"], dtype=np.int64)
    if cache.exists():
        saved = np.load(cache)
        if (str(saved["training_version"].item()) == TRAINING_VERSION
                and np.array_equal(saved["fit_indices"], fit)
                and np.array_equal(saved["held_indices"], held)):
            print(f"V42 fold {fold}: resumed evidence-conditioned features", flush=True)
            return fit, held, saved["fit_features"], saved["held_features"]
    model = SuicideRiskMultiTaskModel()
    model.load_state_dict(torch.load(V20 / f"inner_fold{fold}_model.pt",
                                     map_location="cpu", weights_only=True))
    model.backbone.encoder.gradient_checkpointing_disable()
    model.to(device).eval()

    def collect(indices, description):
        rows = []
        for batch in tqdm(_loader(dataset, indices, False), desc=description):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            with torch.autocast(device_type="cuda", enabled=True):
                hidden = model.backbone(ids, mask).float()
                document = model.pooling(hidden, mask)
                start, end = model.evidence_head(hidden)
                features = _evidence_features(hidden, mask, start, end, document)
            rows.append(features.cpu().numpy())
        return np.vstack(rows)

    fit_features = collect(fit, f"V42 fold {fold} fit evidence features")
    held_features = collect(held, f"V42 fold {fold} OOF evidence features")
    np.savez_compressed(cache, training_version=TRAINING_VERSION,
                        fit_indices=fit, held_indices=held,
                        fit_features=fit_features, held_features=held_features)
    del model; torch.cuda.empty_cache()
    return fit, held, fit_features, held_features


def _pipeline(c_value, balanced):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=float(c_value), class_weight="balanced" if balanced else None,
                           max_iter=3000, solver="lbfgs"),
    )


def _ordinal_fit_probability(train_x, train_y, held_x, c_value, balanced):
    cumulative = []
    for boundary in range(3):
        model = _pipeline(c_value, balanced).fit(train_x, (train_y > boundary).astype(int))
        cumulative.append(model.predict_proba(held_x)[:, 1])
    cumulative = np.minimum.accumulate(np.column_stack(cumulative), axis=1)
    probability = np.column_stack((
        1. - cumulative[:, 0], cumulative[:, 0] - cumulative[:, 1],
        cumulative[:, 1] - cumulative[:, 2], cumulative[:, 2],
    )).clip(1e-7, 1.)
    return probability / probability.sum(1, keepdims=True)


def _metric(truth, prediction, evidence, indices):
    indices = np.asarray(indices, dtype=np.int64)
    risk = float(f1_score(truth[indices], prediction[indices], average="weighted",
                          zero_division=0))
    phrase_values = evidence[np.arange(len(prediction)), prediction]
    phrase = float(phrase_values[indices].mean())
    return risk, phrase, task1_score(risk, phrase), phrase_values


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("V42 evidence-conditioned model requires CUDA")
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

    specifications = [(kind, c, balanced)
                      for kind in ("nominal", "ordinal")
                      for c in (.0001, .0003, .001, .003, .01)
                      for balanced in (False, True)]
    probabilities = {spec: np.zeros((len(records), 4), dtype=np.float32)
                     for spec in specifications}
    device = torch.device("cuda")
    feature_dimension = None
    for fold in range(4):
        fit, held, fit_x, held_x = _extract_fold(fold, dataset, device)
        feature_dimension = int(fit_x.shape[1])
        positions = [position[int(index)] for index in held]
        for kind, c_value, balanced in specifications:
            if kind == "nominal":
                model = _pipeline(c_value, balanced).fit(fit_x, labels[fit])
                value = model.predict_proba(held_x)
            else:
                value = _ordinal_fit_probability(
                    fit_x, labels[fit], held_x, c_value, balanced
                )
            probabilities[(kind, c_value, balanced)][positions] = value
        print(f"V42 fold {fold}: fitted {len(specifications)} evidence heads", flush=True)

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

    weights = (.05, .10, .20, .35, .50)
    crossfit = np.zeros(len(truth), dtype=np.int64); selections = []; folds = []
    for fold in range(4):
        fit_positions = np.flatnonzero(membership != fold)
        held_positions = np.flatnonzero(membership == fold)
        choices = []
        for spec in specifications:
            for weight in weights:
                blended = ((1. - weight) * old_probability
                           + weight * probabilities[spec])
                prediction = _predict(blended, lexical, parameters, corrections)
                score = _metric(truth, prediction, evidence, fit_positions)
                choices.append((score[2], score[0], spec, weight, prediction))
        _, _, spec, weight, prediction = max(choices, key=lambda row: (row[0], row[1]))
        crossfit[held_positions] = prediction[held_positions]
        old = _metric(truth, baseline, evidence, held_positions)
        new = _metric(truth, crossfit, evidence, held_positions)
        selections.append((spec, weight))
        folds.append({"fold": fold, "posts": int(len(held_positions)),
                      "head": spec[0], "c": spec[1], "balanced": spec[2],
                      "evidence_head_weight": weight,
                      "baseline_risk_f1": old[0], "candidate_risk_f1": new[0],
                      "baseline_task1": old[2], "candidate_task1": new[2]})
        print(f"V42 fold={fold} {spec} weight={weight} "
              f"task1 {old[2]:.6f}->{new[2]:.6f}", flush=True)

    cross = _metric(truth, crossfit, evidence, np.arange(len(truth)))
    production_spec = Counter(spec for spec, _ in selections).most_common(1)[0][0]
    chosen_weights = [weight for _, weight in selections]
    centre = float(np.median(chosen_weights))
    production_weight = min(weights, key=lambda value: abs(value - centre))
    blended = ((1. - production_weight) * old_probability
               + production_weight * probabilities[production_spec])
    fixed_prediction = _predict(blended, lexical, parameters, corrections)
    fixed = _metric(truth, fixed_prediction, evidence, np.arange(len(truth)))

    local_groups = groups[global_indices]; unique = np.unique(local_groups)
    rng = np.random.default_rng(config.SEED + 4242); deltas = []
    for _ in range(4000):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        selected = np.concatenate([np.flatnonzero(local_groups == user) for user in sampled])
        old_risk = f1_score(truth[selected], baseline[selected], average="weighted",
                            zero_division=0)
        new_risk = f1_score(truth[selected], crossfit[selected], average="weighted",
                            zero_division=0)
        deltas.append(task1_score(new_risk, float(cross[3][selected].mean()))
                      - task1_score(old_risk, float(base[3][selected].mean())))
    deltas = np.asarray(deltas)
    bootstrap = {"mean_task1_delta": float(deltas.mean()),
                 "p05_task1_delta": float(np.quantile(deltas, .05)),
                 "p95_task1_delta": float(np.quantile(deltas, .95)),
                 "positive_fraction": float((deltas > 0).mean())}
    adopted = bool(cross[2] >= base[2] + .003 and fixed[2] >= base[2] + .002
                   and bootstrap["positive_fraction"] >= .80)
    payload = {"training_version": TRAINING_VERSION,
               "evaluation_scope": "all 1305 nested OOF posts; user-disjoint feature extraction",
               "feature_dimension": feature_dimension,
               "baseline_v36": {"risk_f1": base[0], "phrase_f1": base[1],
                                "task1": base[2]},
               "crossfit_candidate": {"risk_f1": cross[0], "phrase_f1": cross[1],
                                      "task1": cross[2], "folds": folds},
               "fixed_production_diagnostic": {
                   "head": production_spec[0], "c": production_spec[1],
                   "balanced": production_spec[2], "evidence_head_weight": production_weight,
                   "risk_f1": fixed[0], "phrase_f1": fixed[1], "task1": fixed[2]},
               "user_cluster_bootstrap": bootstrap, "adopted": adopted}
    RESULTS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    CALIBRATION.write_text(json.dumps({"training_version": TRAINING_VERSION,
        "adopted": adopted, "head": production_spec[0], "c": production_spec[1],
        "balanced": production_spec[2], "evidence_head_weight": production_weight,
        "crossfit_task1": cross[2], "baseline_task1": base[2]}, indent=2),
        encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    return payload


if __name__ == "__main__":
    main()
