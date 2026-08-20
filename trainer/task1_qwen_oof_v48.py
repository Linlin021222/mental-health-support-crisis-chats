"""Four-fold user-disjoint confirmation of the Qwen QLoRA expert (V48)."""
from __future__ import annotations

import json
import gc

import numpy as np
import torch
from sklearn.metrics import f1_score
from transformers import AutoTokenizer

from analyze_task1_oof_risk_v36 import CACHE as V36_CACHE, _evidence_matrix, _predict as v36_predict
from configs.config import config
from inference.task1_evidence_v4 import correct_risk_only
from preprocess.preprocess import load_train_data
from trainer.task1_atomic_v25 import _load_records
from trainer.task1_large_v46 import _risk_probability
from trainer.task1_qwen_lora_v47 import (
    EPOCHS, MODEL_NAME, PREDECLARED_WEIGHT, PromptDataset,
    _infer, _model_and_tokenizer, _train,
)
from utils.seed import seed_everything
from utils.task1_metric import task1_score


OUTPUT = config.OUTPUT_DIR / "task1_qwen_oof_v48"
RESULTS = OUTPUT / "results.json"
CALIBRATION = OUTPUT / "calibration.json"
FULL_ADAPTER = OUTPUT / "full_adapter"
TRAINING_VERSION = "task1-qwen25-3b-fourfold-oof-v48"


def _metric(truth, prediction, evidence, indices):
    indices = np.asarray(indices, dtype=np.int64)
    risk = float(f1_score(truth[indices], prediction[indices],
                          average="weighted", zero_division=0))
    phrase_values = evidence[np.arange(len(prediction)), prediction]
    phrase = float(phrase_values[indices].mean())
    return risk, phrase, task1_score(risk, phrase), phrase_values


def _corrections(texts):
    return np.asarray([[correct_risk_only(text, risk) for risk in range(4)]
                       for text in texts], dtype=np.int64)


def _train_fold(fold, dataset, labels, membership, global_indices):
    prediction_file = OUTPUT / f"fold{fold}_probabilities.npz"
    adapter_dir = OUTPUT / f"fold{fold}_adapter"
    fit = global_indices[membership != fold]
    held = global_indices[membership == fold]
    if prediction_file.exists():
        saved = np.load(prediction_file)
        if (str(saved["training_version"].item()) == TRAINING_VERSION
                and np.array_equal(saved["global_indices"], held)):
            print(f"V48 fold {fold}: resumed {len(held)} probabilities", flush=True)
            return held, saved["probabilities"]
    seed_everything(config.SEED + 4800 + fold)
    model, tokenizer = _model_and_tokenizer()
    history = _train(model, dataset, fit, labels)
    probability = _infer(model, dataset, held, f"V48 fold {fold} held users")
    model.save_pretrained(adapter_dir); tokenizer.save_pretrained(adapter_dir)
    np.savez_compressed(prediction_file, training_version=TRAINING_VERSION,
                        global_indices=held, probabilities=probability)
    (OUTPUT / f"fold{fold}_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return held, probability


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("V48 requires CUDA")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame = load_train_data().reset_index(drop=True)
    labels = frame.risk_label.to_numpy(dtype=np.int64)
    groups = frame.anon_user_id.astype(str).to_numpy()
    records, membership_map, _ = _load_records()
    global_indices = np.asarray([int(row["global_index"]) for row in records])
    membership = np.asarray([membership_map[int(index)] for index in global_indices])
    truth = labels[global_indices]
    texts = [str(row["text"]) for row in records]
    local_groups = groups[global_indices]

    # Tokenization does not require a temporary 3B model.  Avoiding that model
    # is important on an 8 GB GPU because PEFT/bitsandbytes objects can retain
    # CUDA allocations until cyclic garbage collection runs.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    dataset = PromptDataset(frame.text.astype(str).tolist(), labels, tokenizer)

    qwen = np.zeros((len(records), 4), dtype=np.float32)
    position = {int(index): local for local, index in enumerate(global_indices)}
    folds = []
    for fold in range(4):
        held, probability = _train_fold(
            fold, dataset, labels, membership, global_indices,
        )
        local = np.asarray([position[int(index)] for index in held])
        qwen[local] = probability
        folds.append({"fold": fold, "train_posts": int((membership != fold).sum()),
                      "valid_posts": int(len(held))})

    saved = np.load(V36_CACHE, allow_pickle=True)
    names = saved["names"].tolist(); decisions = saved["decisions"]
    v36 = json.loads((config.OUTPUT_DIR / "task1_oof_risk_v36" / "calibration.json")
                     .read_text(encoding="utf-8"))
    lexical = decisions[names.index(v36["expert"])]
    transformer = np.vstack([row["old_probability"] for row in records])
    corrections = _corrections(texts)
    baseline = v36_predict(transformer, lexical, v36, corrections)
    candidate = _risk_probability(
        texts, transformer, qwen, lexical, v36, PREDECLARED_WEIGHT,
    )
    evidence = _evidence_matrix(records)
    base = _metric(truth, baseline, evidence, np.arange(len(truth)))
    new = _metric(truth, candidate, evidence, np.arange(len(truth)))
    for row in folds:
        held = np.flatnonzero(membership == row["fold"])
        old_fold = _metric(truth, baseline, evidence, held)
        new_fold = _metric(truth, candidate, evidence, held)
        row.update({"baseline_task1": old_fold[2], "candidate_task1": new_fold[2],
                    "changed_predictions": int(np.sum(
                        baseline[held] != candidate[held]))})
        print(f"V48 fold={row['fold']} task1 "
              f"{old_fold[2]:.6f}->{new_fold[2]:.6f}", flush=True)

    unique = np.unique(local_groups); rng = np.random.default_rng(config.SEED + 4848)
    deltas = []
    for _ in range(4000):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        selected = np.concatenate([np.flatnonzero(local_groups == user)
                                   for user in sampled])
        old_risk = f1_score(truth[selected], baseline[selected],
                            average="weighted", zero_division=0)
        new_risk = f1_score(truth[selected], candidate[selected],
                            average="weighted", zero_division=0)
        deltas.append(task1_score(new_risk, float(new[3][selected].mean()))
                      - task1_score(old_risk, float(base[3][selected].mean())))
    deltas = np.asarray(deltas)
    bootstrap = {"mean_delta": float(deltas.mean()),
                 "p05_delta": float(np.quantile(deltas, .05)),
                 "p95_delta": float(np.quantile(deltas, .95)),
                 "positive_fraction": float((deltas > 0).mean())}
    stage_one = json.loads((config.OUTPUT_DIR / "task1_qwen_lora_v47" / "results.json")
                           .read_text(encoding="utf-8"))
    adopted = bool(new[2] >= base[2] + .003
                   and bootstrap["positive_fraction"] >= .80
                   and stage_one.get("promising_for_full_oof", False))

    full_history = None
    if adopted:
        seed_everything(config.SEED + 4899)
        model, tokenizer = _model_and_tokenizer()
        full_history = _train(model, dataset, np.arange(len(frame)), labels)
        model.save_pretrained(FULL_ADAPTER); tokenizer.save_pretrained(FULL_ADAPTER)
        del model
        gc.collect()
        torch.cuda.empty_cache()
        print(f"V48 full adapter: {FULL_ADAPTER}", flush=True)

    payload = {"training_version": TRAINING_VERSION, "model": MODEL_NAME,
        "method": {"folds": 4, "epochs_per_fold": EPOCHS,
            "fixed_qwen_weight": PREDECLARED_WEIGHT,
            "parameter_selection_on_oof": False},
        "evaluation_scope": "all 1305 posts are predicted by a user-disjoint QLoRA fold",
        "baseline_v36": {"risk_f1": base[0], "phrase_f1": base[1],
                         "task1": base[2]},
        "candidate": {"risk_f1": new[0], "phrase_f1": new[1],
            "task1": new[2], "changed_predictions": int(np.sum(baseline != candidate)),
            "folds": folds},
        "user_cluster_bootstrap": bootstrap, "full_history": full_history,
        "adopted": adopted}
    RESULTS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    CALIBRATION.write_text(json.dumps({"training_version": TRAINING_VERSION,
        "adopted": adopted, "qwen_weight": PREDECLARED_WEIGHT,
        "crossfit_task1": new[2], "baseline_task1": base[2]}, indent=2),
        encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    return payload


if __name__ == "__main__":
    main()
