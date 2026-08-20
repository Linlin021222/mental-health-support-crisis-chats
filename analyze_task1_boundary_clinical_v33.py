"""Cross-validated composition of V29 boundaries and V32 clinical scores."""
from __future__ import annotations

import json

import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from transformers import AutoTokenizer

from analyze_task1_atomic_refine_v26 import _predict as _boundary_predict
from baseline import _post_phrase_f1
from configs.config import config
from inference.task1_evidence_v4 import load_evidence_calibration
from preprocess.preprocess import load_train_data
from trainer.task1_atomic_v25 import (
    AtomicEvidenceModel, _baseline_evidence, _bootstrap, _build_examples,
    _infer, _load_records,
)
from trainer.task1_clinical_reranker_v32 import (
    OUTPUT as V32_OUTPUT, _clinical_rows, _make_model, _predictions, _score,
)
from trainer.task1_evidence_reranker_v13 import _pool_records
from trainer.task1_oof_stack_v20 import _parameter_grid
from utils.task1_metric import task1_score


OUTPUT = config.OUTPUT_DIR / "task1_boundary_clinical_v33"
RESULTS = OUTPUT / "results.json"
CALIBRATION = OUTPUT / "calibration.json"
TRAINING_VERSION = "task1-boundary-clinical-composition-v33"


def _load_clinical_scores(posts, indices, fold_name, tokenizer, device):
    rows = _clinical_rows(posts, indices, training=False)
    dataset_model = _make_model().to(device)
    dataset_model.load_state_dict(torch.load(
        V32_OUTPUT / f"{fold_name}_model.pt", map_location=device, weights_only=True
    ))
    from trainer.task1_evidence_reranker_v13 import PairDataset
    dataset = PairDataset(rows, tokenizer)
    scores = _score(dataset_model, dataset, device, f"V33 {fold_name} clinical")
    del dataset_model, dataset; torch.cuda.empty_cache()
    return rows, scores


def _boundary_for_fold(frame, global_indices, local_indices, posts, risks,
                       boundary, tokenizer, device, fold):
    if fold == 3:
        saved = torch.load(
            config.OUTPUT_DIR / "task1_atomic_refine_v26" / "atomic_outputs.pt",
            map_location="cpu", weights_only=False,
        )
        outputs = saved["inner_outputs"]
    else:
        examples = _build_examples(frame, global_indices, tokenizer, training=False)
        checkpoint = torch.load(
            config.OUTPUT_DIR / "task1_boundary_crossval_v29" / f"fold{fold}_model.pt",
            map_location="cpu", weights_only=False,
        )
        model = AtomicEvidenceModel().to(device)
        model.load_state_dict(checkpoint["state_dict"])
        outputs = _infer(model, examples, device, f"V33 fold {fold} atomic")
        del model, examples; torch.cuda.empty_cache()
    baselines = {int(g): list(posts[int(l)]["baseline_evidence"])
                 for g, l in zip(global_indices, local_indices)}
    predictions = _boundary_predict(
        frame, global_indices, outputs, risks, baselines, boundary
    )
    return predictions


def _scores(predictions, gold):
    return np.asarray([_post_phrase_f1(prediction, target)
                       for prediction, target in zip(predictions, gold)], dtype=np.float32)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("V33 requires CUDA")
    OUTPUT.mkdir(parents=True, exist_ok=True); device = torch.device("cuda")
    frame = load_train_data().reset_index(drop=True)
    labels = frame.risk_label.to_numpy(); groups = frame.anon_user_id.astype(str).to_numpy()
    outer_train, _ = next(StratifiedGroupKFold(
        n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED
    ).split(np.zeros(len(frame)), labels, groups))
    raw_records, membership, _ = _load_records()
    record_by_global = {int(row["global_index"]): row for row in raw_records}
    global_to_local = {int(row["global_index"]): local
                       for local, row in enumerate(raw_records)}
    posts = _pool_records(raw_records, use_truth=False)
    evidence_calibration = load_evidence_calibration()
    for post, record in zip(posts, raw_records):
        post["baseline_evidence"] = _baseline_evidence(record, evidence_calibration)
    boundary = json.loads((
        config.OUTPUT_DIR / "task1_atomic_refine_v26" / "results.json"
    ).read_text(encoding="utf-8"))["selected"]
    atomic_tokenizer = AutoTokenizer.from_pretrained(
        config.MODEL_NAME, use_fast=True, local_files_only=True
    )
    clinical_tokenizer = AutoTokenizer.from_pretrained(
        config.FACTOR_MODEL_NAME, use_fast=True, local_files_only=True
    )

    def prepare(fold):
        global_idx = np.asarray([i for i in outer_train if membership[int(i)] == fold])
        local_idx = np.asarray([global_to_local[int(i)] for i in global_idx])
        risks = {int(i): int(record_by_global[int(i)]["risk"]) for i in global_idx}
        boundary_predictions = _boundary_for_fold(
            frame, global_idx, local_idx, posts, risks, boundary,
            atomic_tokenizer, device, fold,
        )
        # Parameter-grid helpers read the baseline from each post.
        for local, evidence in zip(local_idx, boundary_predictions):
            posts[int(local)]["baseline_evidence"] = list(evidence)
        rows, scores = _load_clinical_scores(
            posts, local_idx, "calibration" if fold == 3 else f"fold{fold}",
            clinical_tokenizer, device,
        )
        return global_idx, local_idx, boundary_predictions, rows, scores

    cal_global, cal_local, cal_boundary, cal_rows, cal_scores = prepare(3)
    selected = _parameter_grid(
        posts, cal_rows, cal_scores, cal_local, evidence_calibration
    )[0]
    policy = {key: selected[key] for key in ("mode", "topk", "threshold", "gate")}
    print(f"V33 fixed fold3 policy: {selected}", flush=True)

    all_global = []; all_raw = []; all_boundary = []; all_new = []; all_risks = []; folds = []
    for fold in (0, 1, 2):
        global_idx, local_idx, boundary_predictions, rows, scores = prepare(fold)
        candidate_predictions = _predictions(posts, rows, scores, local_idx, policy)
        raw_predictions = [_baseline_evidence(record_by_global[int(i)], evidence_calibration)
                           for i in global_idx]
        gold = [list(frame.iloc[int(i)].evidence) for i in global_idx]
        raw = _scores(raw_predictions, gold); boundary_values = _scores(boundary_predictions, gold)
        new = _scores(candidate_predictions, gold)
        folds.append({"fold": fold, "posts": int(len(global_idx)),
                      "raw_phrase_f1": float(raw.mean()),
                      "boundary_phrase_f1": float(boundary_values.mean()),
                      "candidate_phrase_f1": float(new.mean()),
                      "delta_vs_boundary": float(new.mean() - boundary_values.mean()),
                      "improved_vs_boundary": int((new > boundary_values).sum()),
                      "worsened_vs_boundary": int((new < boundary_values).sum())})
        print(f"V33 fold={fold} boundary {boundary_values.mean():.6f} -> {new.mean():.6f}", flush=True)
        all_global.extend(map(int, global_idx)); all_raw.extend(raw); all_boundary.extend(boundary_values)
        all_new.extend(new); all_risks.extend(int(record_by_global[int(i)]["risk"]) for i in global_idx)

    order = np.argsort(all_global); indices = np.asarray(all_global)[order]
    raw = np.asarray(all_raw, dtype=np.float32)[order]
    boundary_values = np.asarray(all_boundary, dtype=np.float32)[order]
    new = np.asarray(all_new, dtype=np.float32)[order]
    risks = np.asarray(all_risks, dtype=np.int64)[order]
    risk_f1 = float(f1_score(labels[indices], risks, average="weighted", zero_division=0))
    raw_task = task1_score(risk_f1, float(raw.mean()))
    boundary_task = task1_score(risk_f1, float(boundary_values.mean()))
    new_task = task1_score(risk_f1, float(new.mean()))
    bootstrap = _bootstrap(groups[indices], boundary_values, new)
    adopted = bool(new_task >= boundary_task + .003 and bootstrap["positive_fraction"] >= .80)
    payload = {"training_version": TRAINING_VERSION,
        "evaluation_scope": "combined policy selected fold3; untouched user folds0-2",
        "fixed_boundary": boundary, "selected_clinical_policy": policy,
        "folds": folds,
        "raw_baseline": {"risk_f1": risk_f1, "phrase_f1": float(raw.mean()), "task1": raw_task},
        "boundary_baseline": {"risk_f1": risk_f1, "phrase_f1": float(boundary_values.mean()),
                              "task1": boundary_task},
        "candidate": {"risk_f1": risk_f1, "phrase_f1": float(new.mean()), "task1": new_task,
                      "improved_posts": int((new > boundary_values).sum()),
                      "worsened_posts": int((new < boundary_values).sum())},
        "user_cluster_bootstrap": bootstrap, "adopted": adopted}
    RESULTS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    CALIBRATION.write_text(json.dumps({"training_version": TRAINING_VERSION,
        "adopted": adopted, "policy": policy, "crossvalidated_task1": new_task},
        indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True); return payload


if __name__ == "__main__":
    main()
