"""Fold-0 shared text/label cross-encoder for long-tailed Task 2."""
import json
import random
import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification, AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

from configs.config import config
from inference.factor_nli import TRAIN_NLI_FILE, _entailment_index, _rank_decode
from preprocess.preprocess import load_train_data
from trainer.factor_cv import OOF_FILE
from utils.seed import seed_everything


OUTPUT_DIR = config.OUTPUT_DIR / "factor_cross_encoder"
CHECKPOINT = OUTPUT_DIR / "fold0_model.pt"
PREDICTION_FILE = OUTPUT_DIR / "fold0_valid.npz"
RESULT_FILE = OUTPUT_DIR / "fold0_ablation.json"

# Confusable labels supply more useful negatives than 24-way uniform sampling.
CONFUSION_GROUPS = [
    [0, 1, 2, 16], [3, 4, 5], [6, 7], [8, 11, 12, 14, 15],
    [9, 13, 17], [10, 19], [18], [20, 21, 22, 23],
]


class PairDataset(Dataset):
    def __init__(self, texts, pairs):
        self.texts, self.pairs = texts, pairs
    def __len__(self):
        return len(self.pairs)
    def __getitem__(self, index):
        row, label, target, weight = self.pairs[index]
        return self.texts[row], label, target, weight


def _training_pairs(targets, seed):
    rng = random.Random(seed)
    supports = targets.sum(0).clip(min=1)
    maximum = float(supports.max())
    pairs = []
    for row, truth in enumerate(targets):
        positives = np.flatnonzero(truth).tolist()
        negatives = set()
        for positive in positives:
            for group in CONFUSION_GROUPS:
                if positive in group:
                    negatives.update(x for x in group if not truth[x])
        negative_target = max(6, 2 * len(positives))
        available = [x for x in range(config.NUM_FACTORS) if not truth[x] and x not in negatives]
        rng.shuffle(available)
        negatives.update(available[:max(0, negative_target - len(negatives))])
        negatives = list(negatives)
        rng.shuffle(negatives)
        negatives = negatives[:negative_target]
        for label in positives:
            # Tail positive pairs matter more, but cap their leverage to avoid
            # a handful of noisy annotations destabilising the shared encoder.
            weight = min(4.0, float((maximum / supports[label]) ** 0.35))
            pairs.append((row, label, 1, weight))
        pairs.extend((row, label, 0, 1.0) for label in negatives)
    rng.shuffle(pairs)
    return pairs


def _collator(tokenizer):
    def collate(rows):
        texts, labels, targets, weights = zip(*rows)
        hypotheses = [config.FACTOR_NLI_HYPOTHESES[int(x)] for x in labels]
        encoded = tokenizer(
            list(texts), hypotheses, padding=True, truncation="only_first",
            max_length=config.FACTOR_NLI_MAX_LENGTH, return_tensors="pt",
        )
        encoded["targets"] = torch.tensor(targets, dtype=torch.long)
        encoded["weights"] = torch.tensor(weights, dtype=torch.float)
        return encoded
    return collate


@torch.no_grad()
def _predict(model, tokenizer, texts, device):
    model.eval()
    entailment = _entailment_index(model)
    probabilities = np.zeros((len(texts), config.NUM_FACTORS), dtype=np.float32)
    batch_size = config.FACTOR_CROSS_ENCODER_BATCH_SIZE * 2
    for label, hypothesis in enumerate(tqdm(
        config.FACTOR_NLI_HYPOTHESES, desc="cross-encoder validation"
    )):
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            encoded = tokenizer(
                batch, [hypothesis] * len(batch), padding=True,
                truncation="only_first", max_length=config.FACTOR_NLI_MAX_LENGTH,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(**encoded).logits.float()
            probabilities[start:start + len(batch), label] = (
                torch.softmax(logits, dim=-1)[:, entailment].cpu().numpy()
            )
    return probabilities


def train_cross_encoder_fold0():
    seed_everything(config.SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = load_train_data().reset_index(drop=True)
    targets = np.vstack(frame.factor_vector.to_numpy()).astype(np.int8)
    train_idx, valid_idx = next(StratifiedGroupKFold(
        n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED
    ).split(np.zeros(len(frame)), frame.risk_label, frame.anon_user_id))
    train_targets = targets[train_idx]
    local_pairs = _training_pairs(train_targets, config.SEED)
    train_texts = frame.text.iloc[train_idx].tolist()
    tokenizer = AutoTokenizer.from_pretrained(config.FACTOR_NLI_MODEL_NAME, use_fast=True)
    loader = DataLoader(
        PairDataset(train_texts, local_pairs),
        batch_size=config.FACTOR_CROSS_ENCODER_BATCH_SIZE, shuffle=True,
        collate_fn=_collator(tokenizer), num_workers=0,
    )
    device = torch.device(config.DEVICE)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.FACTOR_NLI_MODEL_NAME, dtype=torch.float32,
    ).to(device)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    entailment = _entailment_index(model)
    optimizer = AdamW(model.parameters(), lr=config.FACTOR_CROSS_ENCODER_LR,
                      weight_decay=config.WEIGHT_DECAY)
    update_steps = int(np.ceil(len(loader) / config.FACTOR_CROSS_ENCODER_ACCUMULATION))
    total_steps = update_steps * config.FACTOR_CROSS_ENCODER_EPOCHS
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, max(1, int(total_steps * config.WARMUP_RATIO)), total_steps
    )
    scaler = torch.amp.GradScaler("cuda", enabled=config.FP16)
    prevalence = train_targets.mean(0)
    valid_texts = frame.text.iloc[valid_idx].tolist()
    valid_targets = targets[valid_idx]
    best, best_epoch, best_probability = -1.0, 0, None
    epochs = []
    print(f"cross-encoder fold0: train_posts={len(train_idx)} pairs={len(local_pairs)} "
          f"valid_posts={len(valid_idx)}")
    for epoch in range(1, config.FACTOR_CROSS_ENCODER_EPOCHS + 1):
        model.train(); optimizer.zero_grad(set_to_none=True); losses = []
        for step, batch in enumerate(tqdm(loader, desc=f"cross-encoder epoch {epoch}"), 1):
            targets_binary = batch.pop("targets").to(device)
            weights = batch.pop("weights").to(device)
            batch = {key: value.to(device) for key, value in batch.items()}
            # Positive means the entailment class; negative means the other class.
            class_targets = torch.where(
                targets_binary > 0,
                torch.full_like(targets_binary, entailment),
                torch.full_like(targets_binary, 1 - entailment),
            )
            with torch.autocast(device_type=device.type, enabled=config.FP16):
                logits = model(**batch).logits
                raw = torch.nn.functional.cross_entropy(
                    logits, class_targets, reduction="none"
                )
                loss = (raw * weights).mean() / config.FACTOR_CROSS_ENCODER_ACCUMULATION
            scaler.scale(loss).backward()
            losses.append(float(loss.detach()) * config.FACTOR_CROSS_ENCODER_ACCUMULATION)
            if (step % config.FACTOR_CROSS_ENCODER_ACCUMULATION == 0
                    or step == len(loader)):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
                old_scale = scaler.get_scale()
                scaler.step(optimizer); scaler.update()
                # GradScaler may skip an overflowing first update.  Advancing
                # the scheduler on a skipped optimizer step triggers PyTorch's
                # ordering warning and shortens the effective warmup.
                if scaler.get_scale() >= old_scale:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
        probability = _predict(model, tokenizer, valid_texts, device)
        prediction = _rank_decode(probability, prevalence, 1.0)
        score = f1_score(valid_targets, prediction, average="macro", zero_division=0)
        loss = float(np.mean(losses))
        epochs.append({"epoch": epoch, "loss": loss, "macro_f1": float(score)})
        print(f"cross-encoder epoch={epoch} loss={loss:.4f} macro_f1={score:.4f}")
        if score > best:
            best, best_epoch, best_probability = float(score), epoch, probability.copy()
            torch.save(model.state_dict(), CHECKPOINT)

    old = np.load(OOF_FILE)
    base = (config.FACTOR_SEMANTIC_MODEL_WEIGHT * old["semantic"][valid_idx]
            + config.FACTOR_CPU_ENSEMBLE_WEIGHT * old["cpu"][valid_idx])
    systems = {
        "cross_encoder": best_probability,
        "old_neural_cpu": base,
    }
    if TRAIN_NLI_FILE.exists():
        nli = np.load(TRAIN_NLI_FILE)["probabilities"][valid_idx]
        systems["old_neural_cpu_nli"] = 0.70 * base + 0.30 * nli
        for weight in (0.20, 0.30, 0.40, 0.50):
            systems[f"old_base_cross_{weight:.2f}"] = (
                (1.0 - weight) * base + weight * best_probability
            )
            systems[f"old_base_nli_cross_{weight:.2f}"] = (
                (1.0 - weight) * (0.70 * base + 0.30 * nli)
                + weight * best_probability
            )
    comparisons = {}
    for name, probability in systems.items():
        ratio = 1.0 if ("nli" in name or "cross" in name) else 1.10
        comparisons[name] = float(f1_score(
            valid_targets, _rank_decode(probability, prevalence, ratio),
            average="macro", zero_division=0,
        ))
    payload = {"best_epoch": best_epoch, "epochs": epochs,
               "pair_count": len(local_pairs), "comparisons": comparisons}
    np.savez_compressed(PREDICTION_FILE, probabilities=best_probability,
                        valid_indices=valid_idx)
    RESULT_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return payload


if __name__ == "__main__":
    train_cross_encoder_fold0()
