"""Calibrate one high-confidence evidence addition, validate on V29 folds."""
from __future__ import annotations

import json

import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from transformers import AutoTokenizer

from analyze_task1_atomic_refine_v26 import _refine_one
from baseline import _post_phrase_f1
from configs.config import config
from inference.task1_evidence_v4 import load_evidence_calibration
from preprocess.preprocess import load_train_data
from trainer.task1_atomic_v25 import (
    HARD_CUE, AtomicEvidenceModel, _atomic_candidates, _baseline_evidence,
    _bootstrap, _build_examples, _infer, _load_records, _normalise,
)
from utils.task1_metric import task1_score


OUTPUT = config.OUTPUT_DIR / "task1_boundary_augment_v30"
RESULTS = OUTPUT / "results.json"
CALIBRATION = OUTPUT / "calibration.json"
TRAINING_VERSION = "task1-boundary-high-precision-augment-v30"


def _allowed_risk(risk, policy):
    if policy == "all":
        return risk > 0
    if policy == "behavior_attempt":
        return risk >= config.RISK_LABELS["Behavior"]
    if policy == "behavior":
        return risk == config.RISK_LABELS["Behavior"]
    return False


def _augment(text, baseline, atomic, risk, parameters):
    refined = _refine_one(text, baseline, atomic, parameters["boundary"])
    if parameters["mode"] == "boundary_only":
        return refined
    if len(refined) > parameters["maximum_baseline_count"] or not _allowed_risk(risk, parameters["risk_policy"]):
        return refined
    existing = [_normalise(value) for value in refined]
    for score, _, _, phrase in atomic:
        normal = _normalise(phrase)
        if score < parameters["addition_gate"]:
            continue
        if parameters["cue_only"] and not HARD_CUE.search(phrase):
            continue
        if any(normal in old or old in normal for old in existing):
            continue
        return refined + [phrase]
    return refined


def _predictions(frame, indices, outputs, risks, baselines, parameters):
    result = []
    boundary = parameters["boundary"]
    for index in map(int, indices):
        risk = int(risks[index])
        if risk == 0:
            result.append([]); continue
        text = str(frame.iloc[index].text)
        atomic = _atomic_candidates(
            text, outputs.get(index, []), boundary["token_threshold"],
            boundary["sentence_threshold"], boundary["max_tokens"],
        )
        result.append(_augment(text, baselines[index], atomic, risk, parameters))
    return result


def _score(frame, indices, predictions):
    return np.asarray([
        _post_phrase_f1(prediction, list(frame.iloc[int(index)].evidence))
        for prediction, index in zip(predictions, indices)
    ], dtype=np.float32)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("V30 requires CUDA")
    OUTPUT.mkdir(parents=True, exist_ok=True); device = torch.device("cuda")
    frame = load_train_data().reset_index(drop=True)
    labels = frame.risk_label.to_numpy(); groups = frame.anon_user_id.astype(str).to_numpy()
    outer_train, _ = next(StratifiedGroupKFold(
        n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED
    ).split(np.zeros(len(frame)), labels, groups))
    records, membership, _ = _load_records()
    record_by_index = {int(row["global_index"]): row for row in records}
    boundary = json.loads((
        config.OUTPUT_DIR / "task1_atomic_refine_v26" / "results.json"
    ).read_text(encoding="utf-8"))["selected"]
    calibration = load_evidence_calibration()
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME, use_fast=True, local_files_only=True)

    # Fold 3 alone selects the augmentation policy.
    cal_idx = np.asarray([i for i in outer_train if membership[int(i)] == 3])
    saved = torch.load(
        config.OUTPUT_DIR / "task1_atomic_refine_v26" / "atomic_outputs.pt",
        map_location="cpu", weights_only=False,
    )
    cal_outputs = saved["inner_outputs"]
    cal_baseline = {int(i): _baseline_evidence(record_by_index[int(i)], calibration) for i in cal_idx}
    cal_risk = {int(i): int(record_by_index[int(i)]["risk"]) for i in cal_idx}
    candidates = [{"mode": "boundary_only", "boundary": boundary,
                   "addition_gate": 1.0, "maximum_baseline_count": 0,
                   "risk_policy": "all", "cue_only": True}]
    for gate in (0.60, 0.70, 0.80, 0.90):
        for maximum in (0, 1, 2):
            for risk_policy in ("all", "behavior_attempt", "behavior"):
                for cue_only in (True, False):
                    candidates.append({"mode": "append_one", "boundary": boundary,
                                       "addition_gate": gate,
                                       "maximum_baseline_count": maximum,
                                       "risk_policy": risk_policy, "cue_only": cue_only})
    rows = []
    for parameters in candidates:
        prediction = _predictions(frame, cal_idx, cal_outputs, cal_risk, cal_baseline, parameters)
        values = _score(frame, cal_idx, prediction)
        rows.append({**parameters, "phrase_f1": float(values.mean())})
    rows.sort(key=lambda row: row["phrase_f1"], reverse=True)
    selected = rows[0]
    print(f"V30 selected on fold3: {selected}", flush=True)

    # Fixed-policy validation on the three folds never used for selection.
    all_indices = []; old_values = []; new_values = []; all_risks = []; folds = []
    for fold in (0, 1, 2):
        valid_idx = np.asarray([i for i in outer_train if membership[int(i)] == fold])
        examples = _build_examples(frame, valid_idx, tokenizer, training=False)
        checkpoint = torch.load(
            config.OUTPUT_DIR / "task1_boundary_crossval_v29" / f"fold{fold}_model.pt",
            map_location="cpu", weights_only=False,
        )
        model = AtomicEvidenceModel().to(device); model.load_state_dict(checkpoint["state_dict"])
        outputs = _infer(model, examples, device, f"V30 validation fold {fold}")
        baselines = {int(i): _baseline_evidence(record_by_index[int(i)], calibration) for i in valid_idx}
        risks = {int(i): int(record_by_index[int(i)]["risk"]) for i in valid_idx}
        old_predictions = _predictions(frame, valid_idx, outputs, risks, baselines,
            {"mode": "boundary_only", "boundary": boundary, "addition_gate": 1.0,
             "maximum_baseline_count": 0, "risk_policy": "all", "cue_only": True})
        new_predictions = _predictions(frame, valid_idx, outputs, risks, baselines, selected)
        old = _score(frame, valid_idx, old_predictions); new = _score(frame, valid_idx, new_predictions)
        folds.append({"fold": fold, "posts": int(len(valid_idx)),
                      "boundary_phrase_f1": float(old.mean()),
                      "augmented_phrase_f1": float(new.mean()),
                      "delta": float(new.mean() - old.mean()),
                      "improved": int((new > old).sum()), "worsened": int((new < old).sum())})
        all_indices.extend(map(int, valid_idx)); old_values.extend(old); new_values.extend(new)
        all_risks.extend(risks[int(i)] for i in valid_idx)
        del model, examples, outputs; torch.cuda.empty_cache()
    order = np.argsort(all_indices); indices = np.asarray(all_indices)[order]
    old = np.asarray(old_values, dtype=np.float32)[order]
    new = np.asarray(new_values, dtype=np.float32)[order]
    risks = np.asarray(all_risks, dtype=np.int64)[order]
    risk_f1 = float(f1_score(labels[indices], risks, average="weighted", zero_division=0))
    old_task = task1_score(risk_f1, old.mean()); new_task = task1_score(risk_f1, new.mean())
    bootstrap = _bootstrap(groups[indices], old, new)
    adopted = bool(new_task >= old_task + .003 and bootstrap["positive_fraction"] >= .80)
    payload = {"training_version": TRAINING_VERSION,
        "evaluation_scope": "augmentation selected fold3; validated folds0-2 over fixed V29 boundary",
        "selected": selected, "calibration_top10": rows[:10], "folds": folds,
        "boundary_baseline": {"risk_f1": risk_f1, "phrase_f1": float(old.mean()), "task1": old_task},
        "candidate": {"risk_f1": risk_f1, "phrase_f1": float(new.mean()), "task1": new_task,
                      "improved_posts": int((new > old).sum()), "worsened_posts": int((new < old).sum())},
        "user_cluster_bootstrap": bootstrap, "adopted": adopted}
    RESULTS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    CALIBRATION.write_text(json.dumps({"training_version": TRAINING_VERSION,
        "adopted": adopted, "crossvalidated_task1": new_task, "selected": selected}, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True); return payload


if __name__ == "__main__":
    main()
