"""Strict gate for local Qwen7B counterfactual augmentation (Task 1 V56).

Only source posts belonging to the outer training users are rewritten.  The
untouched outer validation users, decoder parameters, lexical calibration and
ensemble weight are inherited from the accepted V52 experiment.
"""
from __future__ import annotations

import copy
import json

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoTokenizer

from analyze_task1_lexical_v11 import _lexical_experts, _transformer_probability
from baseline import _post_phrase_f1
from configs.config import config
from datasets.cache_builder import _all_occurrences
from datasets.dataset import SuicideRiskDataset
from inference.task1_evidence_v4 import correct_risk_only
from models.multitask_model import SuicideRiskMultiTaskModel, get_optimizer_parameters
from preprocess.preprocess import load_train_data
from trainer.task1_atomic_v25 import _load_records
from trainer.task1_local_counterfactual_v56 import OUTPUT, SYNTHETIC_FILE
from trainer.task1_qwen7b_verbalizer_v53 import _ensemble_prediction
from trainer.task1_rationale_augment_v52 import _augment
from trainer.task1_seed_evidence_v28 import _decode
from trainer.task1_seed_ensemble_v14 import _collect_seed, _criterion
from trainer.train import _loader, _move
from utils.seed import seed_everything
from utils.task1_metric import task1_score


RESULTS = OUTPUT / "strict_results.json"
CHECKPOINT = OUTPUT / "strict_model.pt"
PREDICTIONS = OUTPUT / "strict_predictions.pt"
FULL_CHECKPOINT = OUTPUT / "full_model.pt"
FULL_MANIFEST = OUTPUT / "full_manifest.json"
TRAINING_VERSION = "task1-local-qwen7b-counterfactual-v56"
SEED = 565656
SYNTHETIC_REPEAT = 2
FIXED_RISK_WEIGHT = .10


def _encode(text, evidence, risk, factors, row_id, tokenizer):
    encoded = tokenizer(
        text, max_length=config.MAX_LENGTH, stride=config.STRIDE,
        truncation=True, return_overflowing_tokens=True,
        return_offsets_mapping=True, padding="max_length",
    )
    count = min(len(encoded["input_ids"]), config.MAX_CHUNKS)
    spans = [span for phrase in evidence for span in _all_occurrences(text, phrase)]
    ids, masks, starts, ends, tokens, offsets = [], [], [], [], [], []
    for chunk in range(config.MAX_CHUNKS):
        if chunk < count:
            chunk_offsets = [tuple(pair) for pair in encoded["offset_mapping"][chunk]]
            input_ids = encoded["input_ids"][chunk]
            attention = encoded["attention_mask"][chunk]
            start = np.zeros(config.MAX_LENGTH, dtype=np.float32)
            end = np.zeros(config.MAX_LENGTH, dtype=np.float32)
            token = np.zeros(config.MAX_LENGTH, dtype=np.float32)
            for left, right in spans:
                positions = [i for i, (a, b) in enumerate(chunk_offsets)
                             if b > left and a < right and b > a]
                if positions:
                    start[positions[0]] = 1.; end[positions[-1]] = 1.
                    token[positions] = 1.
        else:
            input_ids = [tokenizer.pad_token_id] * config.MAX_LENGTH
            attention = [0] * config.MAX_LENGTH
            chunk_offsets = [(0, 0)] * config.MAX_LENGTH
            start = end = token = np.zeros(config.MAX_LENGTH, dtype=np.float32)
        ids.append(input_ids); masks.append(attention); starts.append(start)
        ends.append(end); tokens.append(token); offsets.append(chunk_offsets)
    if not sum(float(np.asarray(item).sum()) for item in tokens):
        raise ValueError(f"Synthetic evidence was not token-aligned: {row_id}")
    return {
        "row_id": row_id, "anon_user_id": row_id, "text": text,
        "offset_mapping": offsets, "evidence": list(evidence),
        "input_ids": torch.tensor(ids, dtype=torch.long),
        "attention_mask": torch.tensor(masks, dtype=torch.long),
        "start_labels": torch.tensor(np.asarray(starts), dtype=torch.float32),
        "end_labels": torch.tensor(np.asarray(ends), dtype=torch.float32),
        "token_labels": torch.tensor(np.asarray(tokens), dtype=torch.float32),
        "risk_label": torch.tensor(int(risk), dtype=torch.long),
        "factor_vector": torch.tensor(factors, dtype=torch.float32),
    }


def _append_synthetic(dataset):
    payload = json.loads(SYNTHETIC_FILE.read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(
        config.MODEL_NAME, use_fast=True, local_files_only=True,
    )
    original = len(dataset.data); records = []
    for number, row in enumerate(payload["accepted"]):
        records.append(_encode(
            row["post"], [row["evidence"]], row["target_risk"],
            row["factor_vector"], f"v56_cf_{number}", tokenizer,
        ))
    for repeat in range(SYNTHETIC_REPEAT):
        for record in records:
            item = copy.copy(record)
            item["row_id"] = f"{record['row_id']}__repeat_{repeat}"
            dataset.data.append(item)
    indices = np.arange(original, len(dataset.data), dtype=int)
    return indices, len(records)


def _train(dataset, indices, valid_idx, labels, force=False):
    if CHECKPOINT.exists() and PREDICTIONS.exists() and not force:
        saved = torch.load(PREDICTIONS, map_location="cpu", weights_only=False)
        if np.array_equal(saved["valid_idx"], valid_idx):
            print(f"{TRAINING_VERSION} strict training resumed", flush=True)
            return saved["rows"], saved["history"]
    device = torch.device("cuda"); seed_everything(SEED)
    model = SuicideRiskMultiTaskModel().to(device)
    optimizer = AdamW(get_optimizer_parameters(model), weight_decay=config.WEIGHT_DECAY)
    criterion = _criterion(dataset, indices, labels, device)
    loader = _loader(dataset, indices, True)
    scaler = torch.amp.GradScaler("cuda", enabled=True); history = []
    for epoch in range(1, 4):
        model.train(); optimizer.zero_grad(set_to_none=True); losses = []
        progress = tqdm(loader, desc=f"{TRAINING_VERSION} strict epoch {epoch}/3")
        for step, batch in enumerate(progress, 1):
            batch = _move(batch, device)
            with torch.autocast(device_type="cuda", enabled=True):
                loss = criterion(
                    model(batch["input_ids"], batch["attention_mask"]), batch,
                )["loss"] / config.GRADIENT_ACCUMULATION
            scaler.scale(loss).backward()
            losses.append(float(loss.detach()) * config.GRADIENT_ACCUMULATION)
            if step % config.GRADIENT_ACCUMULATION == 0 or step == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update()
                optimizer.zero_grad(set_to_none=True)
        history.append({"epoch": epoch, "loss": float(np.mean(losses))})
        print(f"{TRAINING_VERSION} epoch={epoch} loss={history[-1]['loss']:.5f}", flush=True)
    torch.save(model.state_dict(), CHECKPOINT)
    rows = _collect_seed(model, _loader(dataset, valid_idx, False), device)
    torch.save({"valid_idx": valid_idx, "rows": rows, "history": history}, PREDICTIONS)
    del model, optimizer; torch.cuda.empty_cache()
    return rows, history


def _metric(truth, risk, phrase):
    risk_f1 = float(f1_score(truth, risk, average="weighted", zero_division=0))
    phrase_f1 = float(np.mean(phrase))
    return {"risk_f1": risk_f1, "phrase_f1": phrase_f1,
            "task1": task1_score(risk_f1, phrase_f1)}


def train_full(force=False):
    """Train an accepted counterfactual expert on every labelled post."""
    if not torch.cuda.is_available():
        raise RuntimeError("Counterfactual full training requires CUDA")
    strict = json.loads(RESULTS.read_text(encoding="utf-8"))
    if not strict.get("adopted", False):
        raise RuntimeError("The strict counterfactual branch did not pass its gate")
    if FULL_CHECKPOINT.exists() and not force:
        print(f"Counterfactual full checkpoint already exists: {FULL_CHECKPOINT}", flush=True)
        return FULL_CHECKPOINT
    dataset = SuicideRiskDataset(config.CACHE_DIR / "train_cache.pt")
    original = len(dataset); all_idx = np.arange(original, dtype=int)
    rationale_idx, rationale_count = _augment(dataset, all_idx)
    synthetic_idx, unique_synthetic = _append_synthetic(dataset)
    expanded_idx = np.concatenate((rationale_idx, synthetic_idx))
    labels = np.asarray([int(row["risk_label"]) for row in dataset.data])
    device = torch.device("cuda"); seed_everything(SEED)
    model = SuicideRiskMultiTaskModel().to(device)
    optimizer = AdamW(get_optimizer_parameters(model), weight_decay=config.WEIGHT_DECAY)
    criterion = _criterion(dataset, expanded_idx, labels, device)
    loader = _loader(dataset, expanded_idx, True)
    scaler = torch.amp.GradScaler("cuda", enabled=True); history = []
    print(f"Counterfactual full: originals={original}, rationale={rationale_count}, "
          f"synthetic_unique={unique_synthetic}, total={len(expanded_idx)}", flush=True)
    for epoch in range(1, config.FULL_TRAIN_EPOCHS + 1):
        model.train(); optimizer.zero_grad(set_to_none=True); losses = []
        progress = tqdm(loader, desc=f"Counterfactual full epoch {epoch}/{config.FULL_TRAIN_EPOCHS}")
        for step, batch in enumerate(progress, 1):
            batch = _move(batch, device)
            with torch.autocast(device_type="cuda", enabled=True):
                loss = criterion(
                    model(batch["input_ids"], batch["attention_mask"]), batch,
                )["loss"] / config.GRADIENT_ACCUMULATION
            scaler.scale(loss).backward()
            losses.append(float(loss.detach()) * config.GRADIENT_ACCUMULATION)
            if step % config.GRADIENT_ACCUMULATION == 0 or step == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update()
                optimizer.zero_grad(set_to_none=True)
        history.append({"epoch": epoch, "loss": float(np.mean(losses))})
        print(f"Counterfactual full epoch={epoch} loss={history[-1]['loss']:.5f}", flush=True)
    torch.save(model.state_dict(), FULL_CHECKPOINT)
    FULL_MANIFEST.write_text(json.dumps({
        "training_version": TRAINING_VERSION, "original_posts": original,
        "rationale_views": rationale_count, "synthetic_unique": unique_synthetic,
        "synthetic_repeat": SYNTHETIC_REPEAT, "training_views": len(expanded_idx),
        "epochs": config.FULL_TRAIN_EPOCHS, "seed": SEED,
        "risk_weight": FIXED_RISK_WEIGHT, "evidence_weight": .10,
        "history": history,
    }, indent=2), encoding="utf-8")
    print(f"Counterfactual full checkpoint ready: {FULL_CHECKPOINT}", flush=True)
    del model, optimizer; torch.cuda.empty_cache()
    return FULL_CHECKPOINT


def main(force=False):
    if not torch.cuda.is_available():
        raise RuntimeError("V56 strict training requires CUDA")
    if not SYNTHETIC_FILE.exists():
        raise FileNotFoundError("Run --mode task1-local-cf-v56-generate first")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    dataset = SuicideRiskDataset(config.CACHE_DIR / "train_cache.pt")
    frame = load_train_data().reset_index(drop=True)
    labels = frame.risk_label.to_numpy(dtype=np.int64)
    groups = frame.anon_user_id.astype(str).to_numpy()
    train_idx, valid_idx = next(StratifiedGroupKFold(
        n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED,
    ).split(np.zeros(len(frame)), labels, groups))
    rationale_idx, rationale_count = _augment(dataset, train_idx)
    synthetic_idx, unique_synthetic = _append_synthetic(dataset)
    expanded_idx = np.concatenate((rationale_idx, synthetic_idx))
    extended_labels = np.asarray([int(row["risk_label"]) for row in dataset.data])
    print(f"{TRAINING_VERSION} strict: originals={len(train_idx)}, rationale={rationale_count}, "
          f"synthetic_unique={unique_synthetic}, synthetic_repeated={len(synthetic_idx)}, "
          f"total={len(expanded_idx)}", flush=True)
    rows, history = _train(dataset, expanded_idx, valid_idx, extended_labels, force)

    _, _, raw = _load_records(); records = raw["records"]
    transformer = _transformer_probability(dataset, valid_idx, records)
    v52_saved = torch.load(
        config.OUTPUT_DIR / "task1_rationale_augment_v52" / "strict_predictions.pt",
        map_location="cpu", weights_only=False,
    )
    v52_probability = np.vstack([row["probability"] for row in v52_saved["rows"]])
    new_probability = np.vstack([row["probability"] for row in rows])
    v36 = json.loads((config.OUTPUT_DIR / "task1_oof_risk_v36" / "calibration.json")
                     .read_text(encoding="utf-8"))
    lexical = _lexical_experts(frame, train_idx, valid_idx)[v36["expert"]]
    texts = frame.text.iloc[valid_idx].astype(str).tolist()
    baseline_risk = _ensemble_prediction(
        texts, transformer, v52_probability, new_probability, lexical, v36, 0.0,
    )
    candidate_risk = _ensemble_prediction(
        texts, transformer, v52_probability, new_probability, lexical, v36,
        FIXED_RISK_WEIGHT,
    )

    v18 = json.loads((config.OUTPUT_DIR / "task1_candidate_v18" / "results.json")
                     .read_text(encoding="utf-8"))
    v35 = json.loads((config.OUTPUT_DIR / "task1_oof_decoder_v35" / "calibration.json")
                     .read_text(encoding="utf-8"))
    params = (v35["parameters_by_predicted_risk"] if v35.get("adopted", False)
              else v18["evidence_parameters_by_predicted_risk"])
    seed2 = torch.load(
        config.OUTPUT_DIR / "task1_seed_ensemble_v14" / "seed2_valid.pt",
        map_location="cpu", weights_only=False,
    )["rows"]
    starts = [.8 * record["start"] + .2 * seed["start"]
              for record, seed in zip(records, seed2)]
    ends = [.8 * record["end"] + .2 * seed["end"]
            for record, seed in zip(records, seed2)]
    augmented_starts = [.7 * record["start"] + .2 * seed["start"] + .1 * new["start"]
                        for record, seed, new in zip(records, seed2, rows)]
    augmented_ends = [.7 * record["end"] + .2 * seed["end"] + .1 * new["end"]
                      for record, seed, new in zip(records, seed2, rows)]
    baseline_evidence = _decode(records, baseline_risk, starts, ends, params)
    candidate_evidence = _decode(records, candidate_risk, starts, ends, params)
    augmented_evidence = _decode(
        records, candidate_risk, augmented_starts, augmented_ends, params,
    )
    gold = [list(frame.iloc[int(index)].evidence) for index in valid_idx]
    base_phrase = np.asarray([_post_phrase_f1(a, b)
                              for a, b in zip(baseline_evidence, gold)])
    candidate_phrase = np.asarray([_post_phrase_f1(a, b)
                                   for a, b in zip(candidate_evidence, gold)])
    augmented_phrase = np.asarray([_post_phrase_f1(a, b)
                                   for a, b in zip(augmented_evidence, gold)])
    truth = labels[valid_idx]
    baseline = _metric(truth, baseline_risk, base_phrase)
    candidate = _metric(truth, candidate_risk, candidate_phrase)
    evidence_candidate = _metric(truth, candidate_risk, augmented_phrase)

    unique = np.unique(groups[valid_idx]); rng = np.random.default_rng(SEED)
    deltas = []; evidence_deltas = []
    for _ in range(4000):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        positions = np.concatenate([
            np.flatnonzero(groups[valid_idx] == user) for user in sampled
        ])
        old_risk = f1_score(truth[positions], baseline_risk[positions],
                            average="weighted", zero_division=0)
        new_risk = f1_score(truth[positions], candidate_risk[positions],
                            average="weighted", zero_division=0)
        deltas.append(task1_score(new_risk, candidate_phrase[positions].mean())
                      - task1_score(old_risk, base_phrase[positions].mean()))
        evidence_deltas.append(
            task1_score(new_risk, augmented_phrase[positions].mean())
            - task1_score(old_risk, base_phrase[positions].mean())
        )
    deltas = np.asarray(deltas)
    bootstrap = {
        "mean_delta": float(deltas.mean()),
        "p05_delta": float(np.quantile(deltas, .05)),
        "p95_delta": float(np.quantile(deltas, .95)),
        "positive_fraction": float((deltas > 0).mean()),
    }
    evidence_deltas = np.asarray(evidence_deltas)
    evidence_bootstrap = {
        "mean_delta": float(evidence_deltas.mean()),
        "p05_delta": float(np.quantile(evidence_deltas, .05)),
        "p95_delta": float(np.quantile(evidence_deltas, .95)),
        "positive_fraction": float((evidence_deltas > 0).mean()),
    }
    risk_only_adopted = bool(candidate["task1"] >= baseline["task1"] + .003
                             and bootstrap["positive_fraction"] >= .8)
    evidence_adopted = bool(
        evidence_candidate["task1"] >= baseline["task1"] + .003
        and evidence_bootstrap["positive_fraction"] >= .8
    )
    adopted = bool(risk_only_adopted or evidence_adopted)
    payload = {
        "training_version": TRAINING_VERSION,
        "strict_user_isolation": True,
        "augmentation": {
            "accepted_unique": unique_synthetic,
            "repeat": SYNTHETIC_REPEAT,
            "fixed_risk_weight": FIXED_RISK_WEIGHT,
            "rationale_views": rationale_count,
        },
        "history": history, "baseline": baseline,
        "candidate": {
            **candidate,
            "changed_risk": int(np.sum(baseline_risk != candidate_risk)),
            "improved_phrase_posts": int((candidate_phrase > base_phrase).sum()),
            "worsened_phrase_posts": int((candidate_phrase < base_phrase).sum()),
            "confusion": confusion_matrix(truth, candidate_risk,
                                          labels=np.arange(4)).tolist(),
        },
        "fixed_10pct_evidence_candidate": {
            **evidence_candidate,
            "improved_phrase_posts": int((augmented_phrase > base_phrase).sum()),
            "worsened_phrase_posts": int((augmented_phrase < base_phrase).sum()),
            "user_cluster_bootstrap": evidence_bootstrap,
            "adopted": evidence_adopted,
        },
        "risk_only_user_cluster_bootstrap": bootstrap,
        "selected_branch": ("fixed_10pct_evidence" if evidence_adopted
                            else ("risk_only" if risk_only_adopted else None)),
        "adopted": adopted,
    }
    RESULTS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    return payload


if __name__ == "__main__":
    main()
