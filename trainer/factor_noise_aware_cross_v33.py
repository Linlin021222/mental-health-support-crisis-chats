"""Noise-aware reproduction of the accepted V3 prototype cross-encoder.

V31 independently audited a small, pre-selected set of positive post/label
pairs from fold0 training users.  Pairs for which both conservative Qwen
judges found the factor absent are not relabelled or deleted; their positive
loss is reduced to 25%.  Everything else (initial checkpoint, pair sampling,
optimizer and decoder) matches the accepted V3 experiment.
"""
from __future__ import annotations

import json

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_cosine_schedule_with_warmup

from configs.config import config
from inference.factor_nli import _entailment_index, _rank_decode
from preprocess.preprocess import load_train_data
from trainer.factor_cross_encoder_v2 import (
    OLD_DIR, OUTPUT_DIR as V3_DIR, PrototypePairDataset, _collator, _predict,
    _prototype_pairs,
)
from trainer.factor_llm_lexical_v6 import _current_v3_probability
from trainer.factor_sentence_evidence_cv_v27 import _user_bootstrap
from utils.seed import seed_everything


OUTPUT = config.OUTPUT_DIR / "factor_noise_aware_cross_v33"
CHECKPOINT = OUTPUT / "fold0_model.pt"
PREDICTIONS = OUTPUT / "fold0_valid.npz"
RESULTS = OUTPUT / "fold0_results.json"
RATIONALES = config.OUTPUT_DIR / "factor_dual_rationale_v31" / "grounded_rationales.json"
TRAINING_VERSION = "dual-audit-noise-aware-prototype-cross-v33"
ABSENT_POSITIVE_WEIGHT = 0.25
REPLACEMENT_WEIGHT = 0.20
TOPK_RATIO = 1.10


def _audited_absent(train_idx):
    payload = json.loads(RATIONALES.read_text(encoding="utf-8"))
    local = {int(global_row): i for i, global_row in enumerate(train_idx)}
    pairs = set()
    for row in payload["audit"]:
        if row["filter_decision"] != "judge_absent":
            continue
        global_row = int(row["row_index"])
        if global_row not in local:
            raise RuntimeError(f"V31 row {global_row} is outside fold0 training users")
        pairs.add((local[global_row], int(row["label_id"])))
    return pairs


def _macro_auc(truth, probability):
    values = [
        roc_auc_score(truth[:, label], probability[:, label])
        for label in range(config.NUM_FACTORS)
        if np.unique(truth[:, label]).size == 2
    ]
    return float(np.mean(values))


def _train(model, tokenizer, pairs, train_texts, device):
    loader = DataLoader(
        PrototypePairDataset(train_texts, pairs),
        batch_size=config.FACTOR_PROTOTYPE_TRAIN_BATCH_SIZE, shuffle=True,
        collate_fn=_collator(tokenizer), num_workers=0,
    )
    model.gradient_checkpointing_enable(); model.config.use_cache = False
    entailment = _entailment_index(model); other = 1 - entailment
    optimizer = AdamW(model.parameters(), lr=3e-6, weight_decay=config.WEIGHT_DECAY)
    accumulation = config.FACTOR_PROTOTYPE_ACCUMULATION
    updates = int(np.ceil(len(loader) / accumulation))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, max(1, int(updates * .08)), updates
    )
    scaler = torch.amp.GradScaler("cuda", enabled=config.FP16)
    model.train(); optimizer.zero_grad(set_to_none=True); losses = []
    for step, batch in enumerate(tqdm(loader, desc="V33 noise-aware fold0"), 1):
        binary = batch.pop("targets").to(device).float()
        weights = batch.pop("weights").to(device)
        mapping = batch.pop("pair_mapping").to(device)
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.autocast(device_type="cuda", enabled=config.FP16):
            logits = model(**batch).logits
            margin = logits[:, entailment] - logits[:, other]
            pair_margin = torch.stack([
                margin[mapping == pair].max() for pair in range(len(binary))
            ])
            raw = torch.nn.functional.binary_cross_entropy_with_logits(
                pair_margin, binary, reduction="none"
            )
            loss = (raw * weights).mean() / accumulation
        scaler.scale(loss).backward(); losses.append(float(loss.detach()) * accumulation)
        if step % accumulation == 0 or step == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
            old_scale = scaler.get_scale(); scaler.step(optimizer); scaler.update()
            if scaler.get_scale() >= old_scale:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)
    return float(np.mean(losses))


def train_fold0():
    if not torch.cuda.is_available():
        raise RuntimeError("Factor noise-aware V33 requires CUDA")
    print("V33 stage 1/6: loading training data", flush=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame = load_train_data().reset_index(drop=True)
    targets = np.vstack(frame.factor_vector.to_numpy()).astype(np.int8)
    counts = np.vstack(frame.factor_counts.to_numpy()).astype(np.float32)
    risk = frame.risk_label.to_numpy(dtype=np.int64)
    groups = frame.anon_user_id.astype(str).to_numpy()
    train_idx, valid_idx = next(StratifiedGroupKFold(
        n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED,
    ).split(np.zeros(len(frame)), risk, groups))
    audited = _audited_absent(train_idx)
    print(
        f"V33 stage 2/6: fold built ({len(train_idx)} train, {len(valid_idx)} valid); "
        f"audited absent={len(audited)}", flush=True,
    )

    if PREDICTIONS.exists():
        saved = np.load(PREDICTIONS)
        if (str(saved["training_version"]) == TRAINING_VERSION
                and np.array_equal(saved["valid_indices"], valid_idx)):
            probability = saved["probabilities"].astype(np.float32)
            train_loss = float(saved["train_loss"])
            pair_count = int(saved["pair_count"])
            affected_count = int(saved["affected_count"])
            print("V33 fold0: resumed", flush=True)
        else:
            raise RuntimeError("Stale V33 cache")
    else:
        print("V33 stage 3/6: loading hard-negative cache and constructing pairs", flush=True)
        hard = np.load(V3_DIR / "fold0_train_hard.npz")
        if not np.array_equal(hard["train_indices"], train_idx):
            raise RuntimeError("V3 hard-negative cache does not match fold0")
        seed_everything(config.SEED + 100)
        pairs = _prototype_pairs(
            targets[train_idx], counts[train_idx], hard["probabilities"],
            config.SEED + 100,
        )
        adjusted = []
        affected_count = 0
        for row, label, prototype, target, weight in pairs:
            if int(target) == 1 and (int(row), int(label)) in audited:
                weight *= ABSENT_POSITIVE_WEIGHT
                affected_count += 1
            adjusted.append((row, label, prototype, target, weight))
        pair_count = len(adjusted)
        print(
            f"V33 fold0 pairs={pair_count}; audited absent post-labels={len(audited)}; "
            f"downweighted prototype pairs={affected_count}", flush=True,
        )
        print("V33 stage 4/6: loading tokenizer and old fold0 checkpoint", flush=True)
        device = torch.device("cuda")
        tokenizer = AutoTokenizer.from_pretrained(
            config.FACTOR_NLI_MODEL_NAME, use_fast=True, local_files_only=True,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            config.FACTOR_NLI_MODEL_NAME, dtype=torch.float32, local_files_only=True,
        ).to(device)
        model.load_state_dict(torch.load(
            OLD_DIR / "fold0_model.pt", map_location="cpu", weights_only=True
        ))
        print("V33 stage 5/6: training noise-aware cross-encoder", flush=True)
        train_loss = _train(
            model, tokenizer, adjusted,
            frame.text.iloc[train_idx].astype(str).tolist(), device,
        )
        print("V33 stage 6/6: scoring untouched validation users", flush=True)
        probability = _predict(
            model, tokenizer, frame.text.iloc[valid_idx].astype(str).tolist(), device
        )
        torch.save(model.state_dict(), CHECKPOINT)
        np.savez_compressed(
            PREDICTIONS, probabilities=probability, valid_indices=valid_idx,
            train_loss=train_loss, pair_count=pair_count,
            affected_count=affected_count, training_version=TRAINING_VERSION,
        )
        del model
        torch.cuda.empty_cache()

    current, calibration = _current_v3_probability()
    accepted = np.load(V3_DIR / "oof_predictions.npz")["probabilities"][valid_idx]
    component_weight = float(calibration["new_cross_weight"])
    candidate_probability = (
        current[valid_idx] + component_weight * REPLACEMENT_WEIGHT
        * (probability - accepted)
    )
    truth = targets[valid_idx]; prevalence = targets[train_idx].mean(0)
    baseline_prediction = _rank_decode(current[valid_idx], prevalence, TOPK_RATIO)
    candidate_prediction = _rank_decode(candidate_probability, prevalence, TOPK_RATIO)
    baseline = float(f1_score(truth, baseline_prediction, average="macro", zero_division=0))
    candidate = float(f1_score(truth, candidate_prediction, average="macro", zero_division=0))
    bootstrap = _user_bootstrap(
        truth, baseline_prediction, candidate_prediction, groups[valid_idx],
        seed=333333, draws=3000,
    )
    per_label = [{
        "label": config.ID2FACTOR[label], "support": int(truth[:, label].sum()),
        "audited_absent_post_labels": sum(x[1] == label for x in audited),
        "baseline_f1": float(f1_score(
            truth[:, label], baseline_prediction[:, label], zero_division=0
        )),
        "candidate_f1": float(f1_score(
            truth[:, label], candidate_prediction[:, label], zero_division=0
        )),
    } for label in range(config.NUM_FACTORS)]
    payload = {
        "training_version": TRAINING_VERSION,
        "evaluation": "untouched outer user fold0; fixed 20% prototype-component replacement",
        "fixed_policy": {
            "audited_absent_positive_weight": ABSENT_POSITIVE_WEIGHT,
            "prototype_component_replacement": REPLACEMENT_WEIGHT,
            "effective_total_weight": component_weight * REPLACEMENT_WEIGHT,
            "topk_ratio": TOPK_RATIO,
        },
        "audited_absent_post_labels": len(audited),
        "downweighted_prototype_pairs": affected_count,
        "pair_count": pair_count, "train_loss": train_loss,
        "accepted_prototype_auc": _macro_auc(truth, accepted),
        "noise_aware_prototype_auc": _macro_auc(truth, probability),
        "baseline_macro_f1": baseline,
        "candidate_macro_f1": candidate,
        "delta": candidate - baseline,
        "user_cluster_bootstrap": bootstrap,
        "per_label": per_label,
        "promising_for_full_oof": bool(
            candidate >= baseline + .005
            and bootstrap["positive_fraction"] >= .70
            and _macro_auc(truth, probability) >= _macro_auc(truth, accepted)
        ),
        "adopted": False,
    }
    RESULTS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    return payload


if __name__ == "__main__":
    train_fold0()
