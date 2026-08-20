"""Prototype-bank and OOF-hard-negative refinement of the Task 2 cross-encoder."""
from __future__ import annotations

import json
import random

import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_cosine_schedule_with_warmup

from configs.config import config
from inference.factor_nli import _entailment_index, _rank_decode
from preprocess.factor_prototypes import FACTOR_PROTOTYPES
from preprocess.preprocess import load_train_data
from trainer.factor_cross_encoder_ablation import _predict as _old_predict
from utils.seed import seed_everything


OLD_DIR = config.OUTPUT_DIR / "factor_cross_encoder"
OUTPUT_DIR = config.OUTPUT_DIR / "factor_cross_encoder_v2"
OLD_OOF_FILE = OLD_DIR / "oof_predictions.npz"
BASE_OOF_FILE = config.OUTPUT_DIR / "factor_cv" / "factor_oof_predictions.npz"
RESULT_FILE = OUTPUT_DIR / "cv_results.json"
CALIBRATION_FILE = OUTPUT_DIR / "calibration.json"
OOF_FILE = OUTPUT_DIR / "oof_predictions.npz"
TRAINING_VERSION = "prototype-mil-v3"

# Boundaries that are repeatedly confused in the released taxonomy.  A post
# carrying one member but not another becomes a supervised contrastive
# negative for the absent member.
CONFUSION_GROUPS = [
    [0, 3, 4, 16],          # disorder vs mood/hopelessness/cognition
    [1, 7, 14],             # physical vs material/event stress
    [2, 20],                # harmful substance use vs coping activity
    [3, 5, 21],             # hopelessness/self-worth vs positive capital
    [6, 14],                # bad performance vs school-related stress
    [7, 14],                # poverty vs a discrete stressful event
    [8, 12, 15],            # interpersonal violence/family/trauma
    [9, 13, 17],            # own prior suicidality/other person/means
    [10, 11, 19],           # lack of support/social difficulty/support
    [11, 14, 15],           # relationship difficulty/event/trauma
    [14, 15],               # stressful event vs traumatic experience
    [18, 8, 11, 12, 14],    # identity issue vs surrounding consequences
    [19, 20, 21, 22, 23],   # five distinct protective-factor concepts
    [3, 21, 23],            # hopelessness vs hope/meaning
    [22, 23],               # duty/responsibility vs meaning/purpose
]


class PrototypePairDataset(Dataset):
    def __init__(self, texts, pairs):
        self.texts, self.pairs = texts, pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        row, label, prototype, target, weight = self.pairs[index]
        return self.texts[row], label, prototype, target, weight


def _prototype_pairs(targets, factor_counts, hard_scores, seed):
    """Use existing leak-free OOF errors as hard negatives for refinement."""
    rng = random.Random(seed)
    supports = targets.sum(0).clip(min=1)
    maximum = float(supports.max())
    pairs = []
    for row, truth in enumerate(targets):
        positives = np.flatnonzero(truth).tolist()
        false_labels = np.flatnonzero(~truth.astype(bool)).tolist()

        confusion = set()
        for positive in positives:
            for group in CONFUSION_GROUPS:
                if positive in group:
                    confusion.update(label for label in group if not truth[label])
        ranked_false = sorted(false_labels, key=lambda label: float(hard_scores[row, label]), reverse=True)
        hard = set(ranked_false[:max(6, 2 * len(positives))])
        negatives = list(confusion | hard)
        negative_target = max(8, 3 * len(positives))
        if len(negatives) < negative_target:
            remainder = [label for label in false_labels if label not in negatives]
            rng.shuffle(remainder); negatives.extend(remainder[:negative_target - len(negatives)])
        negatives = sorted(set(negatives), key=lambda label: float(hard_scores[row, label]), reverse=True)
        negatives = negatives[:negative_target]

        for label in positives:
            support = int(supports[label])
            if support < 80:
                prototypes = range(len(FACTOR_PROTOTYPES[label]))
            elif support < 250:
                start = rng.randrange(len(FACTOR_PROTOTYPES[label]))
                prototypes = (start, (start + 1) % len(FACTOR_PROTOTYPES[label]))
            else:
                prototypes = (rng.randrange(len(FACTOR_PROTOTYPES[label])),)
            tail_weight = min(5.0, float((maximum / supports[label]) ** 0.40))
            repeats = max(0.0, float(factor_counts[row, label]) - 1.0)
            repeat_boost = 1.0 + config.FACTOR_PROTOTYPE_OCCURRENCE_ALPHA * np.log1p(repeats)
            repeat_boost = min(config.FACTOR_PROTOTYPE_MAX_REPEAT_BOOST, repeat_boost)
            positive_weight = min(
                config.FACTOR_PROTOTYPE_MAX_POSITIVE_WEIGHT,
                tail_weight * repeat_boost,
            )
            for prototype in prototypes:
                pairs.append((row, label, int(prototype), 1, positive_weight))

        hard_set = set(ranked_false[:max(4, len(positives))])
        for label in negatives:
            prototype = rng.randrange(len(FACTOR_PROTOTYPES[label]))
            weight = 1.35 if label in hard_set else 1.0
            pairs.append((row, label, prototype, 0, weight))
    rng.shuffle(pairs)
    return pairs


def _collator(tokenizer):
    def collate(rows):
        texts, labels, prototypes, targets, weights = zip(*rows)
        hypotheses = [FACTOR_PROTOTYPES[int(label)][int(proto)]
                      for label, proto in zip(labels, prototypes)]
        encoded = tokenizer(
            list(texts), hypotheses, padding=True, truncation="only_first",
            max_length=config.FACTOR_NLI_MAX_LENGTH,
            stride=min(128, config.FACTOR_NLI_MAX_LENGTH // 3),
            return_overflowing_tokens=True, return_tensors="pt",
        )
        mapping = encoded.pop("overflow_to_sample_mapping").cpu().numpy()
        selected = []
        for local, (label, prototype, target) in enumerate(zip(labels, prototypes, targets)):
            indices = np.flatnonzero(mapping == local)
            if int(target) > 0:
                maximum = config.FACTOR_PROTOTYPE_TRAIN_MAX_CHUNKS
                if len(indices) > maximum:
                    positions = np.linspace(0, len(indices) - 1, maximum).round().astype(int)
                    indices = indices[positions]
            elif len(indices) > 1:
                # Spread negative supervision across a post without tripling
                # the cost of every negative pair.
                indices = indices[[(int(label) + int(prototype)) % len(indices)]]
            selected.extend(indices.tolist())
        selected = np.asarray(selected, dtype=np.int64)
        selected_mapping = mapping[selected]
        selected_tensor = torch.tensor(selected, dtype=torch.long)
        encoded = {
            key: value.index_select(0, selected_tensor) for key, value in encoded.items()
        }
        encoded["pair_mapping"] = torch.tensor(selected_mapping, dtype=torch.long)
        encoded["targets"] = torch.tensor(targets, dtype=torch.long)
        encoded["weights"] = torch.tensor(weights, dtype=torch.float)
        return encoded
    return collate


@torch.no_grad()
def _predict(model, tokenizer, texts, device):
    """Chunk-aware multiple-instance prediction.

    A factor can occur anywhere in a long Reddit post.  The former shared
    cross-encoder silently truncated after the first 512 tokens; here we keep
    evenly spaced windows (normally first/middle/last) and use the strongest
    entailment window for the post-level hypothesis.
    """
    model.eval()
    entailment = _entailment_index(model)
    result = np.zeros((len(texts), config.NUM_FACTORS), dtype=np.float32)
    batch_size = max(1, config.FACTOR_CROSS_ENCODER_BATCH_SIZE)
    for label, prototypes in enumerate(tqdm(FACTOR_PROTOTYPES, desc="prototype labels")):
        prototype_scores = np.zeros((len(texts), len(prototypes)), dtype=np.float32)
        for prototype_index, hypothesis in enumerate(prototypes):
            for start in range(0, len(texts), batch_size):
                batch = texts[start:start + batch_size]
                encoded = tokenizer(
                    batch, [hypothesis] * len(batch), padding=True,
                    truncation="only_first", max_length=config.FACTOR_NLI_MAX_LENGTH,
                    stride=min(128, config.FACTOR_NLI_MAX_LENGTH // 3),
                    return_overflowing_tokens=True, return_tensors="pt",
                )
                mapping = encoded.pop("overflow_to_sample_mapping").cpu().numpy()
                selected = []
                for local in range(len(batch)):
                    indices = np.flatnonzero(mapping == local)
                    if len(indices) > config.FACTOR_NLI_MAX_CHUNKS:
                        positions = np.linspace(
                            0, len(indices) - 1, config.FACTOR_NLI_MAX_CHUNKS
                        ).round().astype(int)
                        indices = indices[positions]
                    selected.extend(indices.tolist())
                selected = np.asarray(selected, dtype=np.int64)
                selected_mapping = mapping[selected]
                selected_tensor = torch.tensor(selected, dtype=torch.long)
                encoded = {
                    key: value.index_select(0, selected_tensor).to(device)
                    for key, value in encoded.items()
                }
                with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    logits = model(**encoded).logits.float()
                chunk_scores = torch.softmax(logits, dim=-1)[:, entailment].cpu().numpy()
                for local in range(len(batch)):
                    prototype_scores[start + local, prototype_index] = float(
                        chunk_scores[selected_mapping == local].max()
                    )
        # Mean is more robust than max for rare labels and differently worded prototypes.
        result[:, label] = prototype_scores.mean(1)
    return result


def _fold_paths(fold):
    return OUTPUT_DIR / f"fold{fold}_model.pt", OUTPUT_DIR / f"fold{fold}_valid.npz"


def _train_fold(fold, train_idx, valid_idx, frame, targets, factor_counts, tokenizer, device):
    checkpoint, prediction_file = _fold_paths(fold)
    if checkpoint.exists() and prediction_file.exists():
        saved = np.load(prediction_file)
        saved_summary = json.loads(str(saved["summary"]))
        if (np.array_equal(saved["valid_indices"], valid_idx)
                and saved_summary.get("training_version") == TRAINING_VERSION):
            print(f"prototype cross-encoder fold {fold}: resumed")
            return saved["probabilities"].astype(np.float32), saved_summary

    seed_everything(config.SEED + 100 + fold)
    train_targets = targets[train_idx]
    train_counts = factor_counts[train_idx]
    train_texts = frame.text.iloc[train_idx].tolist()
    model = AutoModelForSequenceClassification.from_pretrained(
        config.FACTOR_NLI_MODEL_NAME, dtype=torch.float32, local_files_only=True,
    ).to(device)
    old_checkpoint = OLD_DIR / f"fold{fold}_model.pt"
    if not old_checkpoint.exists():
        raise FileNotFoundError(f"Run --mode factor-cross-cv first: {old_checkpoint}")
    model.load_state_dict(torch.load(old_checkpoint, map_location=device))
    # Hard-negative mining must not use another fold's OOF model: that model
    # may have been trained on this fold's validation users.  Score the
    # training partition with its own old checkpoint instead, and cache the
    # result because this read-only pass is moderately expensive.
    hard_file = OUTPUT_DIR / f"fold{fold}_train_hard.npz"
    old_signature = int(old_checkpoint.stat().st_mtime_ns)
    hard_scores = None
    if hard_file.exists():
        saved_hard = np.load(hard_file)
        if (np.array_equal(saved_hard["train_indices"], train_idx)
                and int(saved_hard["checkpoint_signature"]) == old_signature):
            hard_scores = saved_hard["probabilities"].astype(np.float32)
    if hard_scores is None:
        hard_scores = _old_predict(model, tokenizer, train_texts, device)
        np.savez_compressed(
            hard_file, probabilities=hard_scores, train_indices=train_idx,
            checkpoint_signature=old_signature,
        )
    pairs = _prototype_pairs(
        train_targets, train_counts, hard_scores, config.SEED + 100 + fold
    )
    loader = DataLoader(
        PrototypePairDataset(train_texts, pairs),
        batch_size=config.FACTOR_PROTOTYPE_TRAIN_BATCH_SIZE, shuffle=True,
        collate_fn=_collator(tokenizer), num_workers=0,
    )
    model.gradient_checkpointing_enable(); model.config.use_cache = False
    entailment = _entailment_index(model)
    optimizer = AdamW(model.parameters(), lr=3e-6, weight_decay=config.WEIGHT_DECAY)
    accumulation = config.FACTOR_PROTOTYPE_ACCUMULATION
    updates = int(np.ceil(len(loader) / accumulation))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, max(1, int(updates * 0.08)), updates
    )
    scaler = torch.amp.GradScaler("cuda", enabled=config.FP16)
    model.train(); optimizer.zero_grad(set_to_none=True); losses = []
    for step, batch in enumerate(tqdm(loader, desc=f"prototype fold {fold}"), 1):
        binary = batch.pop("targets").to(device).float()
        weights = batch.pop("weights").to(device)
        pair_mapping = batch.pop("pair_mapping").to(device)
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.autocast(device_type=device.type, enabled=config.FP16):
            chunk_logits = model(**batch).logits
            not_entailment = 1 - entailment
            chunk_margin = chunk_logits[:, entailment] - chunk_logits[:, not_entailment]
            # Multiple-instance learning: a post entails the factor when any
            # selected window supports it. For negatives, the hardest window
            # receives the corrective gradient.
            pair_margin = torch.stack([
                chunk_margin[pair_mapping == pair].max()
                for pair in range(len(binary))
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

    probability = _predict(model, tokenizer, frame.text.iloc[valid_idx].tolist(), device)
    prevalence = train_targets.mean(0)
    score = f1_score(
        targets[valid_idx], _rank_decode(probability, prevalence, 1.0),
        average="macro", zero_division=0,
    )
    summary = {
        "fold": fold, "macro_f1": float(score), "pair_count": len(pairs),
        "training_version": TRAINING_VERSION,
        "train_loss": float(np.mean(losses)), "initialised_from": str(old_checkpoint),
        "chunk_aware_inference": True,
        "max_chunks": int(config.FACTOR_NLI_MAX_CHUNKS),
        "multi_instance_training": True,
        "occurrence_alpha": float(config.FACTOR_PROTOTYPE_OCCURRENCE_ALPHA),
    }
    torch.save(model.state_dict(), checkpoint)
    np.savez_compressed(
        prediction_file, probabilities=probability, valid_indices=valid_idx,
        summary=json.dumps(summary),
    )
    del model, optimizer, scheduler, loader
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return probability, summary


def _evaluate_components(base, old_cross, new_cross, targets, indices, prevalence):
    candidates = []
    for base_weight in np.arange(0.30, 0.61, 0.10):
        remaining = 1.0 - base_weight
        for new_fraction in np.arange(0.25, 1.01, 0.25):
            new_weight = remaining * new_fraction
            old_weight = remaining - new_weight
            probability = (base_weight * base[indices]
                           + old_weight * old_cross[indices]
                           + new_weight * new_cross[indices])
            for ratio in (1.0, 1.10, 1.25):
                score = f1_score(
                    targets[indices], _rank_decode(probability, prevalence, ratio),
                    average="macro", zero_division=0,
                )
                candidates.append({
                    "macro_f1": float(score), "base_weight": float(base_weight),
                    "old_cross_weight": float(old_weight), "new_cross_weight": float(new_weight),
                    "prevalence_ratio": float(ratio),
                })
    return sorted(candidates, key=lambda item: item["macro_f1"], reverse=True)


def train_factor_cross_encoder_v2(only_fold0=False):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = load_train_data().reset_index(drop=True)
    targets = np.vstack(frame.factor_vector.to_numpy()).astype(np.int8)
    factor_counts = np.vstack(frame.factor_counts.to_numpy()).astype(np.float32)
    folds = list(StratifiedGroupKFold(
        n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED
    ).split(np.zeros(len(frame)), frame.risk_label, frame.anon_user_id))
    old_cross = np.load(OLD_OOF_FILE)["probabilities"].astype(np.float32)
    base_saved = np.load(BASE_OOF_FILE)
    base = (config.FACTOR_SEMANTIC_MODEL_WEIGHT * base_saved["semantic"]
            + config.FACTOR_CPU_ENSEMBLE_WEIGHT * base_saved["cpu"])
    tokenizer = AutoTokenizer.from_pretrained(
        config.FACTOR_NLI_MODEL_NAME, use_fast=True, local_files_only=True
    )
    device = torch.device(config.DEVICE)
    new_oof = np.zeros_like(old_cross)
    summaries = []
    selected_folds = folds[:1] if only_fold0 else folds
    for fold, (train_idx, valid_idx) in enumerate(selected_folds):
        probability, summary = _train_fold(
            fold, train_idx, valid_idx, frame, targets, factor_counts, tokenizer, device
        )
        new_oof[valid_idx] = probability; summaries.append(summary)

    if only_fold0:
        train_idx, valid_idx = folds[0]
        grid = _evaluate_components(
            base, old_cross, new_oof, targets, valid_idx, targets[train_idx].mean(0)
        )
        predeclared_probability = (
            0.40 * base[valid_idx] + 0.20 * old_cross[valid_idx]
            + 0.40 * new_oof[valid_idx]
        )
        predeclared_score = f1_score(
            targets[valid_idx],
            _rank_decode(predeclared_probability, targets[train_idx].mean(0), 1.10),
            average="macro", zero_division=0,
        )
        old_score = f1_score(
            targets[valid_idx],
            _rank_decode(0.5 * base[valid_idx] + 0.5 * old_cross[valid_idx], targets[train_idx].mean(0), 1.10),
            average="macro", zero_division=0,
        )
        result = {
            "folds": summaries,
            "old_adopted_fold0": float(old_score),
            "predeclared_prototype_blend_fold0": float(predeclared_score),
            "predeclared_delta": float(predeclared_score - old_score),
            "best_fold0_optimistic": grid[0], "top10": grid[:10],
        }
        (OUTPUT_DIR / "fold0_ablation.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        print(json.dumps(result, indent=2)); return result

    prevalence = targets.mean(0)
    full_grid = _evaluate_components(
        base, old_cross, new_oof, targets, np.arange(len(targets)), prevalence
    )
    crossfit_predictions = np.zeros_like(targets, dtype=bool)
    old_crossfit_predictions = np.zeros_like(targets, dtype=bool)
    crossfit_parameters = []
    for fold, (fit, valid) in enumerate(folds):
        parameter = _evaluate_components(
            base, old_cross, new_oof, targets, fit, targets[fit].mean(0)
        )[0]
        probability = (parameter["base_weight"] * base[valid]
                       + parameter["old_cross_weight"] * old_cross[valid]
                       + parameter["new_cross_weight"] * new_oof[valid])
        crossfit_predictions[valid] = _rank_decode(
            probability, targets[fit].mean(0), parameter["prevalence_ratio"]
        )
        old_crossfit_predictions[valid] = _rank_decode(
            0.5 * base[valid] + 0.5 * old_cross[valid],
            targets[fit].mean(0), 1.10,
        )
        crossfit_parameters.append({"fold": fold, **parameter})
    crossfit_score = f1_score(
        targets, crossfit_predictions, average="macro", zero_division=0
    )
    old_crossfit_score = f1_score(
        targets, old_crossfit_predictions, average="macro", zero_division=0
    )
    old_probability = 0.5 * base + 0.5 * old_cross
    old_score = f1_score(
        targets, _rank_decode(old_probability, prevalence, 1.10),
        average="macro", zero_division=0,
    )
    # Production receives one parameter set, so derive it from the five
    # independently selected folds instead of taking the optimistic full-OOF
    # maximum.  Component medians are stable to one unusual fold.
    production = {
        key: float(np.median([item[key] for item in crossfit_parameters]))
        for key in ("base_weight", "old_cross_weight", "new_cross_weight", "prevalence_ratio")
    }
    component_total = sum(production[key] for key in
                          ("base_weight", "old_cross_weight", "new_cross_weight"))
    for key in ("base_weight", "old_cross_weight", "new_cross_weight"):
        production[key] /= component_total
    production_probability = (
        production["base_weight"] * base
        + production["old_cross_weight"] * old_cross
        + production["new_cross_weight"] * new_oof
    )
    production["macro_f1"] = float(f1_score(
        targets, _rank_decode(production_probability, prevalence,
                              production["prevalence_ratio"]),
        average="macro", zero_division=0,
    ))
    best = full_grid[0]
    adopted = (crossfit_score >= old_crossfit_score + 0.003
               and production["macro_f1"] >= old_score)
    result = {
        "training_version": TRAINING_VERSION,
        "folds": summaries, "old_adopted_full_oof": float(old_score),
        "old_adopted_crossfit_macro_f1": float(old_crossfit_score),
        "prototype_crossfit_macro_f1": float(crossfit_score),
        "crossfit_parameters": crossfit_parameters,
        "production_parameter": production,
        "best_full_oof_optimistic": best, "top10": full_grid[:10], "adopted": adopted,
    }
    RESULT_FILE.write_text(json.dumps(result, indent=2), encoding="utf-8")
    CALIBRATION_FILE.write_text(json.dumps({
        **production, "adopted": adopted,
        "training_version": TRAINING_VERSION,
        "crossfit_macro_f1": float(crossfit_score),
        "old_crossfit_macro_f1": float(old_crossfit_score),
        "prototype_count": [len(x) for x in FACTOR_PROTOTYPES],
        "chunk_aware_inference": True,
        "max_chunks": int(config.FACTOR_NLI_MAX_CHUNKS),
        "multi_instance_training": True,
        "occurrence_alpha": float(config.FACTOR_PROTOTYPE_OCCURRENCE_ALPHA),
        "training_prevalence": prevalence.tolist(),
    }, indent=2), encoding="utf-8")
    np.savez_compressed(
        OOF_FILE, probabilities=new_oof, targets=targets, factor_counts=factor_counts
    )
    print(json.dumps(result, indent=2)); return result


if __name__ == "__main__":
    train_factor_cross_encoder_v2()
