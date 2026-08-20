"""Sentence-to-hard-negative pairwise ranking continuation for Task 2."""
from __future__ import annotations

import json
import math

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from configs.config import config
from inference.factor_nli import _entailment_index, _rank_decode
from preprocess.factor_prototypes import FACTOR_PROTOTYPES
from preprocess.preprocess import load_train_data
from trainer.factor_cross_encoder_v2 import _predict
from trainer.factor_llm_lexical_v6 import _current_v3_probability
from trainer.factor_sentence_evidence_cv_v27 import _user_bootstrap
from utils.seed import seed_everything


OUTPUT = config.OUTPUT_DIR / "factor_pairwise_ranking_v29"
CHECKPOINT = OUTPUT / "fold0_model.pt"
PREDICTIONS = OUTPUT / "fold0_valid.npz"
RESULTS = OUTPUT / "fold0_results.json"
TRAINING_VERSION = "sentence-hard-negative-pairwise-ranking-v29"
SOURCE_DIR = config.OUTPUT_DIR / "factor_cross_encoder_v2"
REPLACEMENT_WEIGHT = 0.20
TOPK_RATIO = 1.10
MARGIN = 0.40
PAIR_BATCH = 3
ACCUMULATION = 4
LEARNING_RATE = 1.5e-6


class PairDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        return self.pairs[index]


def _collator(tokenizer):
    def collate(rows):
        positives, negatives, hypotheses, weights = zip(*rows)
        positive = tokenizer(
            list(positives), list(hypotheses), padding=True,
            truncation="only_first", max_length=256, return_tensors="pt",
        )
        negative = tokenizer(
            list(negatives), list(hypotheses), padding=True,
            truncation="only_first", max_length=256, return_tensors="pt",
        )
        return positive, negative, torch.tensor(weights, dtype=torch.float32)
    return collate


def _evidence_map(path):
    result = {}
    if not path.exists():
        return result
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        selected = row.get("selected") or []
        if selected:
            result[(int(row["row_index"]), row["factor"])] = selected[0]["sentence"]
    return result


def _pairs(frame, targets, train_idx, hard_scores, evidence):
    supports = targets[train_idx].sum(0).clip(min=1)
    maximum = float(supports.max())
    local_of_global = {int(global_row): local for local, global_row in enumerate(train_idx)}
    hard_negative = []
    for label in range(config.NUM_FACTORS):
        absent_local = np.flatnonzero(targets[train_idx, label] == 0)
        order = absent_local[np.argsort(hard_scores[absent_local, label])[::-1]]
        hard_negative.append(order[:min(48, len(order))])

    result = []
    for global_row in train_idx:
        for label in np.flatnonzero(targets[global_row]):
            pool = hard_negative[label]
            if not len(pool):
                continue
            local = local_of_global[int(global_row)]
            negative_local = int(pool[(local * 7 + label * 13) % len(pool)])
            negative_global = int(train_idx[negative_local])
            positive_text = evidence.get(
                (int(global_row), config.ID2FACTOR[label]),
                str(frame.text.iloc[global_row]),
            )
            negative_text = str(frame.text.iloc[negative_global])
            prototype = FACTOR_PROTOTYPES[label][
                (local + label) % len(FACTOR_PROTOTYPES[label])
            ]
            tail_weight = min(3.0, float((maximum / supports[label]) ** .25))
            result.append((positive_text, negative_text, prototype, tail_weight))
    rng = np.random.default_rng(config.SEED + 2900)
    rng.shuffle(result)
    return result


def _train(model, tokenizer, pairs, device):
    # Pairwise ranking only needs the last four transformer layers. Freezing
    # the lower eight preserves the accepted representation and fits two
    # post-definition pairs safely on the 8 GB GPU.
    for parameter in model.parameters():
        parameter.requires_grad = False
    trainable_fragments = (
        "encoder.layer.8.", "encoder.layer.9.", "encoder.layer.10.",
        "encoder.layer.11.", "pooler.", "classifier.",
    )
    for name, parameter in model.named_parameters():
        if any(fragment in name for fragment in trainable_fragments):
            parameter.requires_grad = True
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("Pairwise V29 failed to select trainable DeBERTa layers")
    model.gradient_checkpointing_enable()
    loader = DataLoader(
        PairDataset(pairs), batch_size=PAIR_BATCH, shuffle=True,
        collate_fn=_collator(tokenizer), num_workers=0,
    )
    optimizer = AdamW(trainable, lr=LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    entailment = _entailment_index(model)
    other = 1 - entailment
    model.train(); optimizer.zero_grad(set_to_none=True); losses = []
    for step, (positive, negative, weights) in enumerate(
        tqdm(loader, desc="V29 pairwise ranking"), 1
    ):
        positive = {key: value.to(device) for key, value in positive.items()}
        negative = {key: value.to(device) for key, value in negative.items()}
        weights = weights.to(device)
        with torch.autocast(device_type="cuda", enabled=True):
            positive_logits = model(**positive).logits
            negative_logits = model(**negative).logits
            positive_margin = positive_logits[:, entailment] - positive_logits[:, other]
            negative_margin = negative_logits[:, entailment] - negative_logits[:, other]
            ranking = F.softplus(MARGIN - positive_margin + negative_margin)
            classification = (
                F.softplus(-positive_margin) + F.softplus(negative_margin)
            )
            loss = ((ranking + .20 * classification) * weights).mean() / ACCUMULATION
        scaler.scale(loss).backward()
        losses.append(float(loss.detach()) * ACCUMULATION)
        if step % ACCUMULATION == 0 or step == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, config.GRAD_CLIP)
            scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
    return float(np.mean(losses))


def _macro_auc(truth, probability):
    values = [
        roc_auc_score(truth[:, label], probability[:, label])
        for label in range(config.NUM_FACTORS)
        if np.unique(truth[:, label]).size == 2
    ]
    return float(np.mean(values))


def train_fold0():
    if not torch.cuda.is_available():
        raise RuntimeError("Factor pairwise V29 requires CUDA")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame = load_train_data().reset_index(drop=True)
    targets = np.vstack(frame.factor_vector.to_numpy()).astype(np.int8)
    risk = frame.risk_label.to_numpy(dtype=np.int64)
    groups = frame.anon_user_id.astype(str).to_numpy()
    train_idx, valid_idx = next(StratifiedGroupKFold(
        n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED,
    ).split(np.zeros(len(frame)), risk, groups))

    if PREDICTIONS.exists():
        saved = np.load(PREDICTIONS)
        if (str(saved["training_version"]) == TRAINING_VERSION
                and np.array_equal(saved["valid_indices"], valid_idx)):
            probability = saved["probabilities"].astype(np.float32)
            train_loss = float(saved["train_loss"])
            pair_count = int(saved["pair_count"])
            print("V29 fold0: resumed", flush=True)
        else:
            raise RuntimeError("Stale V29 prediction cache")
    else:
        hard = np.load(SOURCE_DIR / "fold0_train_hard.npz")
        if not np.array_equal(hard["train_indices"], train_idx):
            raise RuntimeError("V29 hard-negative cache does not match fold0")
        evidence = _evidence_map(
            config.OUTPUT_DIR / "factor_sentence_evidence_cv_v27"
            / "fold0_pseudo_evidence.jsonl"
        )
        pairs = _pairs(frame, targets, train_idx, hard["probabilities"], evidence)
        pair_count = len(pairs)
        print(
            f"V29 fold0 pairs={pair_count}; localized positives={len(evidence)}",
            flush=True,
        )
        seed_everything(config.SEED + 2900)
        device = torch.device("cuda")
        tokenizer = AutoTokenizer.from_pretrained(
            config.FACTOR_NLI_MODEL_NAME, use_fast=True, local_files_only=True,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            config.FACTOR_NLI_MODEL_NAME, dtype=torch.float32, local_files_only=True,
        ).to(device)
        model.load_state_dict(torch.load(
            SOURCE_DIR / "fold0_model.pt", map_location="cpu", weights_only=True
        ))
        train_loss = _train(model, tokenizer, pairs, device)
        probability = _predict(
            model, tokenizer,
            frame.text.iloc[valid_idx].astype(str).tolist(), device,
        )
        torch.save(model.state_dict(), CHECKPOINT)
        np.savez_compressed(
            PREDICTIONS, probabilities=probability, valid_indices=valid_idx,
            train_loss=train_loss, pair_count=pair_count,
            training_version=TRAINING_VERSION,
        )
        del model
        torch.cuda.empty_cache()

    current, calibration = _current_v3_probability()
    accepted = np.load(SOURCE_DIR / "oof_predictions.npz")["probabilities"][valid_idx]
    component_weight = float(calibration["new_cross_weight"])
    candidate_probability = (
        current[valid_idx] + component_weight * REPLACEMENT_WEIGHT
        * (probability - accepted)
    )
    prevalence = targets[train_idx].mean(0)
    truth = targets[valid_idx]
    baseline_prediction = _rank_decode(current[valid_idx], prevalence, TOPK_RATIO)
    candidate_prediction = _rank_decode(candidate_probability, prevalence, TOPK_RATIO)
    standalone_prediction = _rank_decode(probability, prevalence, TOPK_RATIO)
    baseline = float(f1_score(
        truth, baseline_prediction, average="macro", zero_division=0
    ))
    candidate = float(f1_score(
        truth, candidate_prediction, average="macro", zero_division=0
    ))
    standalone = float(f1_score(
        truth, standalone_prediction, average="macro", zero_division=0
    ))
    bootstrap = _user_bootstrap(
        truth, baseline_prediction, candidate_prediction, groups[valid_idx],
        seed=292929, draws=2000,
    )
    per_label = [{
        "label": config.ID2FACTOR[label], "support": int(truth[:, label].sum()),
        "baseline_f1": float(f1_score(
            truth[:, label], baseline_prediction[:, label], zero_division=0
        )),
        "candidate_f1": float(f1_score(
            truth[:, label], candidate_prediction[:, label], zero_division=0
        )),
    } for label in range(config.NUM_FACTORS)]
    payload = {
        "training_version": TRAINING_VERSION,
        "evaluation": "untouched outer user fold0; fixed replacement and ratio",
        "pair_count": pair_count, "train_loss": train_loss,
        "fixed_policy": {
            "margin": MARGIN, "prototype_component_replacement": REPLACEMENT_WEIGHT,
            "effective_total_weight": component_weight * REPLACEMENT_WEIGHT,
            "prevalence_ratio": TOPK_RATIO,
        },
        "baseline_macro_f1": baseline,
        "pairwise_standalone_macro_f1": standalone,
        "candidate_macro_f1": candidate,
        "delta": candidate - baseline,
        "accepted_prototype_macro_auc": _macro_auc(truth, accepted),
        "pairwise_prototype_macro_auc": _macro_auc(truth, probability),
        "user_cluster_bootstrap": bootstrap,
        "per_label": per_label,
        "promising_for_full_oof": bool(
            candidate >= baseline + .005
            and bootstrap["positive_fraction"] >= .80
        ),
        "adopted": False,
    }
    RESULTS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    return payload


if __name__ == "__main__":
    train_fold0()
