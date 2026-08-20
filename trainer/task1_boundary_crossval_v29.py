"""Second-stage cross-validation of V26's fixed boundary policy."""
from __future__ import annotations

import json

import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from transformers import AutoTokenizer

from analyze_task1_atomic_refine_v26 import _predict
from baseline import _post_phrase_f1
from configs.config import config
from inference.task1_evidence_v4 import load_evidence_calibration
from preprocess.preprocess import load_train_data
from trainer.task1_atomic_v25 import (
    AtomicEvidenceModel, _baseline_evidence, _bootstrap, _build_examples,
    _infer, _load_records, _train_epochs,
)
from utils.seed import seed_everything
from utils.task1_metric import task1_score


OUTPUT = config.OUTPUT_DIR / "task1_boundary_crossval_v29"
RESULTS = OUTPUT / "results.json"
CALIBRATION = OUTPUT / "calibration.json"
TRAINING_VERSION = "task1-boundary-secondary-crossval-v29"
VALIDATION_FOLDS = (0, 1, 2)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("V29 requires CUDA")
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
    evidence_calibration = load_evidence_calibration()
    tokenizer = AutoTokenizer.from_pretrained(
        config.MODEL_NAME, use_fast=True, local_files_only=True
    )
    all_indices = []; all_baseline = []; all_candidate = []; all_risk = []; fold_rows = []
    for fold in VALIDATION_FOLDS:
        fit_idx = np.asarray([i for i in outer_train if membership[int(i)] != fold])
        valid_idx = np.asarray([i for i in outer_train if membership[int(i)] == fold])
        if set(groups[fit_idx]) & set(groups[valid_idx]):
            raise ValueError(f"V29 fold {fold} user leakage")
        seed_everything(config.SEED + 2900 + fold)
        train_examples = _build_examples(frame, fit_idx, tokenizer, training=True)
        model = AtomicEvidenceModel().to(device)
        history = _train_epochs(model, train_examples, device, epochs=1)
        checkpoint = OUTPUT / f"fold{fold}_model.pt"
        torch.save({"training_version": TRAINING_VERSION, "fold": fold,
                    "state_dict": model.state_dict(), "history": history}, checkpoint)
        valid_examples = _build_examples(frame, valid_idx, tokenizer, training=False)
        outputs = _infer(model, valid_examples, device, f"V29 fold {fold} untouched users")
        baselines = {int(i): _baseline_evidence(record_by_index[int(i)], evidence_calibration)
                     for i in valid_idx}
        risks = {int(i): int(record_by_index[int(i)]["risk"]) for i in valid_idx}
        predictions = _predict(frame, valid_idx, outputs, risks, baselines, boundary)
        old = np.asarray([
            _post_phrase_f1(baselines[int(i)], list(frame.iloc[int(i)].evidence))
            for i in valid_idx
        ], dtype=np.float32)
        new = np.asarray([
            _post_phrase_f1(prediction, list(frame.iloc[int(i)].evidence))
            for prediction, i in zip(predictions, valid_idx)
        ], dtype=np.float32)
        fold_rows.append({
            "fold": fold, "fit_posts": int(len(fit_idx)), "valid_posts": int(len(valid_idx)),
            "baseline_phrase_f1": float(old.mean()), "candidate_phrase_f1": float(new.mean()),
            "phrase_delta": float(new.mean() - old.mean()),
            "improved_posts": int((new > old).sum()), "worsened_posts": int((new < old).sum()),
            "history": history,
        })
        print(f"V29 fold={fold} phrase {old.mean():.6f} -> {new.mean():.6f}", flush=True)
        all_indices.extend(map(int, valid_idx)); all_baseline.extend(old.tolist())
        all_candidate.extend(new.tolist()); all_risk.extend(risks[int(i)] for i in valid_idx)
        del model, train_examples, valid_examples, outputs; torch.cuda.empty_cache()

    order = np.argsort(all_indices); indices = np.asarray(all_indices)[order]
    baseline = np.asarray(all_baseline, dtype=np.float32)[order]
    candidate = np.asarray(all_candidate, dtype=np.float32)[order]
    risk = np.asarray(all_risk, dtype=np.int64)[order]
    risk_f1 = float(f1_score(labels[indices], risk, average="weighted", zero_division=0))
    baseline_task = task1_score(risk_f1, baseline.mean())
    candidate_task = task1_score(risk_f1, candidate.mean())
    bootstrap = _bootstrap(groups[indices], baseline, candidate)
    adopted = bool(candidate_task >= baseline_task + .003 and bootstrap["positive_fraction"] >= .80)
    payload = {
        "training_version": TRAINING_VERSION,
        "evaluation_scope": "V26 parameters fixed on fold3; fresh models and untouched users in folds0-2",
        "fixed_boundary_parameters": boundary,
        "folds": fold_rows,
        "aggregate_baseline": {"risk_f1": risk_f1, "phrase_f1": float(baseline.mean()),
                               "task1": baseline_task},
        "aggregate_candidate": {"risk_f1": risk_f1, "phrase_f1": float(candidate.mean()),
                                "task1": candidate_task,
                                "improved_posts": int((candidate > baseline).sum()),
                                "worsened_posts": int((candidate < baseline).sum())},
        "user_cluster_bootstrap": bootstrap, "adopted": adopted,
    }
    RESULTS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    CALIBRATION.write_text(json.dumps({"training_version": TRAINING_VERSION,
        "adopted": adopted, "crossvalidated_task1": candidate_task,
        "bootstrap_positive_fraction": bootstrap["positive_fraction"],
        "fixed_boundary_parameters": boundary}, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True); return payload


if __name__ == "__main__":
    main()
