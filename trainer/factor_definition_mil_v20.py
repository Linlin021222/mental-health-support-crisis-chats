"""Definition-conditioned latent sentence MIL for Task 2.

This experiment deliberately uses *no* sentence annotations, pseudo evidence,
manual reviews, or teacher-selected sentences.  Every post/label pair is a bag
of sentences and is supervised only by the released post-level factor label.
The NLI cross encoder must therefore learn which sentence, if any, supports a
formal/direct/implicit factor description.
"""
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
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

from configs.config import config
from inference.factor_nli import _entailment_index, _rank_decode
from preprocess.factor_semantic_bank_v15 import FACTOR_CLASSIFIER_PROMPTS
from preprocess.preprocess import load_train_data
from trainer.factor_cross_encoder_v2 import CONFUSION_GROUPS
from trainer.factor_llm_lexical_v6 import _current_v3_probability
from trainer.factor_sentence_evidence_v16 import _score_pairs, _sentences
from utils.seed import seed_everything


OUTPUT = config.OUTPUT_DIR / "factor_definition_mil_v20"
RESULTS = OUTPUT / "fold0_results.json"
CHECKPOINT = OUTPUT / "fold0_model.pt"
VALID_PREDICTIONS = OUTPUT / "fold0_valid.npz"
INITIAL_CHECKPOINT = config.OUTPUT_DIR / "factor_semantic_contrast_v15" / "fold0_model.pt"
HARD_FILE = config.OUTPUT_DIR / "factor_semantic_contrast_v15" / "fold0_train_hard.npz"
ACCEPTED_CROSS = config.OUTPUT_DIR / "factor_cross_encoder_v2" / "oof_predictions.npz"
V17_RESULTS = config.OUTPUT_DIR / "factor_sentence_evidence_v17" / "fold0_results.json"
TRAINING_VERSION = "definition-conditioned-latent-sentence-mil-v20"

# Fixed before looking at V20 validation labels.
PROMPT_INDICES = (0, 1, 2, 4)  # formal, direct, implicit, complete boundary
INFER_PROMPT_INDICES = (0, 4)
MAX_LENGTH = 192
# Dynamic batches keep the total number of sentence/definition sequences under
# a GPU-safe budget.  This is substantially faster than one bag per forward on
# an 8 GB card while preserving an effective batch of roughly 16 bags.
SENTENCE_BUDGET = 22
MAX_BAGS_PER_BATCH = 4
ACCUMULATION = 4
LEARNING_RATE = 1.5e-6
ATTENTION_TEMPERATURE = 0.55
REPEAT_AUX_WEIGHT = 0.10
REPLACEMENT_WEIGHT = 0.20
TOPK_RATIO = 1.10
INFER_BATCH = 32


class _BagDataset(Dataset):
    def __init__(self, bags):
        self.bags = bags

    def __len__(self):
        return len(self.bags)

    def __getitem__(self, index):
        return self.bags[index]


def _build_bags(frame, fit_idx, targets, counts, hard_scores):
    """Create post/definition bags using only outer-training labels.

    ``hard_scores`` affect which *known absent* labels are sampled, never the
    target and never which sentence is positive.  This avoids the error
    propagation of V16--V18's hard pseudo sentence evidence.
    """
    rng = random.Random(config.SEED + 200)
    train_targets = targets[fit_idx]
    supports = train_targets.sum(0).clip(min=1)
    maximum = float(supports.max())
    bags = []
    positive_bags = negative_bags = repeated_bags = 0

    for local_row, global_row in enumerate(fit_idx):
        truth = train_targets[local_row]
        sentences = tuple(_sentences(frame.text.iloc[global_row]))
        positives = np.flatnonzero(truth).tolist()
        absent = np.flatnonzero(~truth.astype(bool)).tolist()

        for label in positives:
            support = int(supports[label])
            if support < 80:
                prompt_indices = PROMPT_INDICES
            elif support < 250:
                start = (int(global_row) + int(label)) % len(PROMPT_INDICES)
                prompt_indices = (
                    PROMPT_INDICES[start],
                    PROMPT_INDICES[(start + 2) % len(PROMPT_INDICES)],
                )
            else:
                prompt_indices = (PROMPT_INDICES[
                    (int(global_row) + int(label)) % len(PROMPT_INDICES)
                ],)
            tail_weight = min(4.0, float((maximum / supports[label]) ** .30))
            annotation_count = max(1, int(round(float(counts[global_row, label]))))
            repeat_weight = min(1.20, 1.0 + .08 * np.log1p(annotation_count - 1))
            for prompt_index in prompt_indices:
                bags.append((sentences, int(label), int(prompt_index), 1,
                             tail_weight * repeat_weight, annotation_count))
                positive_bags += 1
                repeated_bags += int(annotation_count > 1)

        confusion = set()
        for positive in positives:
            for group in CONFUSION_GROUPS:
                if positive in group:
                    confusion.update(label for label in group if not truth[label])
        ranked_absent = sorted(
            absent, key=lambda label: float(hard_scores[local_row, label]), reverse=True,
        )
        hard_count = max(4, 2 * len(positives))
        negatives = set(ranked_absent[:hard_count]) | confusion
        target_count = min(10, max(6, 2 * len(positives)))
        if len(negatives) < target_count:
            remaining = [label for label in absent if label not in negatives]
            rng.shuffle(remaining)
            negatives.update(remaining[:target_count - len(negatives)])
        negatives = sorted(
            negatives, key=lambda label: float(hard_scores[local_row, label]), reverse=True,
        )[:target_count]
        hard_set = set(ranked_absent[:hard_count])
        for label in negatives:
            # The complete boundary description explicitly distinguishes close
            # labels and supplies the cleanest supervision for absent factors.
            weight = 1.25 if label in hard_set or label in confusion else 1.0
            bags.append((sentences, int(label), 4, 0, weight, 0))
            negative_bags += 1

    rng.shuffle(bags)
    return bags, {
        "total": len(bags), "positive": positive_bags,
        "negative": negative_bags, "repeated_positive": repeated_bags,
    }


def _collator(tokenizer):
    def collate(rows):
        premises, hypotheses, mapping = [], [], []
        labels, prompt_indices, targets, weights, repeats = [], [], [], [], []
        for bag_index, (sentences, label, prompt_index, target, weight, repeat) in enumerate(rows):
            hypothesis = FACTOR_CLASSIFIER_PROMPTS[int(label)][int(prompt_index)]
            premises.extend(sentences)
            hypotheses.extend([hypothesis] * len(sentences))
            mapping.extend([bag_index] * len(sentences))
            labels.append(label); prompt_indices.append(prompt_index)
            targets.append(target); weights.append(weight); repeats.append(repeat)
        encoded = tokenizer(
            premises, hypotheses, padding=True, truncation="only_first",
            max_length=MAX_LENGTH, return_tensors="pt",
        )
        encoded["bag_mapping"] = torch.tensor(mapping, dtype=torch.long)
        encoded["targets"] = torch.tensor(targets, dtype=torch.float32)
        encoded["weights"] = torch.tensor(weights, dtype=torch.float32)
        encoded["repeats"] = torch.tensor(repeats, dtype=torch.long)
        return encoded
    return collate


def _sentence_budget_batches(bags):
    batches, current, sentence_count = [], [], 0
    for index, bag in enumerate(bags):
        size = len(bag[0])
        if current and (sentence_count + size > SENTENCE_BUDGET
                        or len(current) >= MAX_BAGS_PER_BATCH):
            batches.append(current)
            current, sentence_count = [], 0
        current.append(index); sentence_count += size
    if current:
        batches.append(current)
    return batches


def _attention_pool(values):
    weights = torch.softmax(values / ATTENTION_TEMPERATURE, dim=0)
    return torch.sum(weights * values)


def _train(model, tokenizer, bags, device):
    seed_everything(config.SEED + 200)
    batch_indices = _sentence_budget_batches(bags)
    print(
        f"Dynamic sentence batches={len(batch_indices)}; "
        f"mean bags/batch={len(bags) / len(batch_indices):.2f}; "
        f"sentence budget={SENTENCE_BUDGET}", flush=True,
    )
    loader = DataLoader(
        _BagDataset(bags), batch_sampler=batch_indices,
        collate_fn=_collator(tokenizer), num_workers=0,
    )
    model.gradient_checkpointing_enable(); model.config.use_cache = False
    entailment = _entailment_index(model); not_entailment = 1 - entailment
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE,
                      weight_decay=config.WEIGHT_DECAY)
    updates = int(np.ceil(len(loader) / ACCUMULATION))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, max(1, int(updates * .08)), max(1, updates),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    model.train(); optimizer.zero_grad(set_to_none=True); losses = []
    for step, batch in enumerate(tqdm(loader, desc="definition MIL fold 0"), 1):
        targets = batch.pop("targets").to(device)
        weights = batch.pop("weights").to(device)
        repeats = batch.pop("repeats").to(device)
        mapping = batch.pop("bag_mapping").to(device)
        inputs = {key: value.to(device) for key, value in batch.items()}
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(**inputs).logits
            margins = logits[:, entailment] - logits[:, not_entailment]
            pooled = torch.stack([
                _attention_pool(margins[mapping == bag])
                for bag in range(len(targets))
            ])
            raw = torch.nn.functional.binary_cross_entropy_with_logits(
                pooled, targets, reduction="none",
            )
            loss = (raw * weights).mean()
            # Duplicate annotations remain a weak salience signal only.  They
            # encourage, but never require, more than one supporting sentence.
            repeat_terms = []
            for bag in range(len(targets)):
                if targets[bag] > .5 and repeats[bag] > 1:
                    bag_margins = margins[mapping == bag]
                    k = min(3, int(repeats[bag].item()), len(bag_margins))
                    if k > 1:
                        repeat_terms.append(torch.nn.functional.softplus(
                            -torch.topk(bag_margins, k).values.mean()
                        ))
            if repeat_terms:
                loss = loss + REPEAT_AUX_WEIGHT * torch.stack(repeat_terms).mean()
            loss = loss / ACCUMULATION
        scaler.scale(loss).backward()
        losses.append(float(loss.detach()) * ACCUMULATION)
        if step % ACCUMULATION == 0 or step == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
            old_scale = scaler.get_scale(); scaler.step(optimizer); scaler.update()
            if scaler.get_scale() >= old_scale:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)
    return float(np.mean(losses))


@torch.no_grad()
def _predict(model, tokenizer, texts, device):
    """Average two definition views after latent attention over all sentences."""
    meta, premises, hypotheses = [], [], []
    for row, text in enumerate(tqdm(texts, desc="build definition MIL validation")):
        for label in range(config.NUM_FACTORS):
            for prompt_index in INFER_PROMPT_INDICES:
                for sentence in _sentences(text):
                    meta.append((row, label, prompt_index))
                    premises.append(sentence)
                    hypotheses.append(FACTOR_CLASSIFIER_PROMPTS[label][prompt_index])
    model.eval(); entailment = _entailment_index(model); not_entailment = 1 - entailment
    margins = np.empty(len(premises), dtype=np.float32)
    for start in tqdm(range(0, len(premises), INFER_BATCH),
                      desc="definition MIL inference"):
        stop = min(start + INFER_BATCH, len(premises))
        encoded = tokenizer(
            premises[start:stop], hypotheses[start:stop], padding=True,
            truncation="only_first", max_length=MAX_LENGTH, return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(**encoded).logits.float()
        margins[start:stop] = (
            logits[:, entailment] - logits[:, not_entailment]
        ).cpu().numpy()
    grouped = {}
    for key, margin in zip(meta, margins):
        grouped.setdefault(key, []).append(float(margin))
    prompt_scores = {}
    for key, values in grouped.items():
        tensor = torch.tensor(values, dtype=torch.float32)
        prompt_scores[key] = float(torch.sigmoid(_attention_pool(tensor)))
    output = np.zeros((len(texts), config.NUM_FACTORS), dtype=np.float32)
    for row in range(len(texts)):
        for label in range(config.NUM_FACTORS):
            output[row, label] = float(np.mean([
                prompt_scores[(row, label, prompt)] for prompt in INFER_PROMPT_INDICES
            ]))
    return output


def train_fold0():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for required in (INITIAL_CHECKPOINT, HARD_FILE, ACCEPTED_CROSS):
        if not required.exists():
            raise FileNotFoundError(f"Required strict-fold artifact is missing: {required}")
    frame = load_train_data().reset_index(drop=True)
    targets = np.vstack(frame.factor_vector.to_numpy()).astype(np.int8)
    counts = np.vstack(frame.factor_counts.to_numpy()).astype(np.float32)
    fit_idx, valid_idx = next(StratifiedGroupKFold(
        n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED,
    ).split(np.zeros(len(frame)), frame.risk_label.to_numpy(),
            frame.anon_user_id.astype(str).to_numpy()))
    hard_saved = np.load(HARD_FILE)
    if not np.array_equal(hard_saved["train_indices"], fit_idx):
        raise RuntimeError("V15 hard-negative cache does not match strict fold 0.")
    hard_scores = hard_saved["probabilities"].astype(np.float32)
    tokenizer = AutoTokenizer.from_pretrained(
        config.FACTOR_NLI_MODEL_NAME, use_fast=True, local_files_only=True,
    )
    device = torch.device(config.DEVICE)

    if CHECKPOINT.exists() and VALID_PREDICTIONS.exists():
        saved = np.load(VALID_PREDICTIONS)
        if (str(saved["training_version"]) != TRAINING_VERSION
                or not np.array_equal(saved["valid_indices"], valid_idx)):
            raise RuntimeError("Existing V20 cache does not match this strict split/version.")
        probability = saved["probabilities"].astype(np.float32)
        train_loss = float(saved["train_loss"])
        bag_summary = json.loads(str(saved["bag_summary"]))
        print("V20 fold 0: resumed cached model and predictions", flush=True)
    else:
        bags, bag_summary = _build_bags(
            frame, fit_idx, targets, counts, hard_scores,
        )
        print(f"Built latent bags: {bag_summary}", flush=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            config.FACTOR_NLI_MODEL_NAME, dtype=torch.float32, local_files_only=True,
        )
        model.load_state_dict(torch.load(
            INITIAL_CHECKPOINT, map_location="cpu", weights_only=True,
        ))
        model = model.to(device)
        train_loss = _train(model, tokenizer, bags, device)
        probability = _predict(
            model, tokenizer, frame.text.iloc[valid_idx].astype(str).tolist(), device,
        )
        torch.save(model.state_dict(), CHECKPOINT)
        np.savez_compressed(
            VALID_PREDICTIONS, probabilities=probability, valid_indices=valid_idx,
            train_loss=train_loss, bag_summary=json.dumps(bag_summary),
            training_version=TRAINING_VERSION,
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    current, calibration = _current_v3_probability()
    accepted = np.load(ACCEPTED_CROSS)["probabilities"][valid_idx].astype(np.float32)
    component_weight = float(calibration["new_cross_weight"])
    candidate = (current[valid_idx] + component_weight * REPLACEMENT_WEIGHT
                 * (probability - accepted))
    truth = targets[valid_idx]
    prevalence = targets[fit_idx].mean(0)
    baseline_pred = _rank_decode(current[valid_idx], prevalence, TOPK_RATIO)
    standalone_pred = _rank_decode(probability, prevalence, TOPK_RATIO)
    candidate_pred = _rank_decode(candidate, prevalence, TOPK_RATIO)
    baseline = float(f1_score(truth, baseline_pred, average="macro", zero_division=0))
    standalone = float(f1_score(
        truth, standalone_pred, average="macro", zero_division=0,
    ))
    score = float(f1_score(truth, candidate_pred, average="macro", zero_division=0))
    per_label = []
    for label, name in enumerate(config.FACTOR_LABELS):
        per_label.append({
            "label": name, "support": int(truth[:, label].sum()),
            "baseline_f1": float(f1_score(
                truth[:, label], baseline_pred[:, label], zero_division=0)),
            "standalone_f1": float(f1_score(
                truth[:, label], standalone_pred[:, label], zero_division=0)),
            "candidate_f1": float(f1_score(
                truth[:, label], candidate_pred[:, label], zero_division=0)),
        })
    v17 = None
    if V17_RESULTS.exists():
        v17 = json.loads(V17_RESULTS.read_text(encoding="utf-8")).get("candidate_macro_f1")
    payload = {
        "training_version": TRAINING_VERSION,
        "strict_split": {"train_rows": len(fit_idx), "valid_rows": len(valid_idx),
                         "user_disjoint": True},
        "supervision": {
            "post_level_factor_labels_only": True,
            "manual_sentence_labels": False,
            "pseudo_sentence_labels": False,
            "teacher_selected_sentences": False,
            "boundary_review_v19_used": False,
        },
        "method": {
            "definition_views_train": list(PROMPT_INDICES),
            "definition_views_inference": list(INFER_PROMPT_INDICES),
            "latent_sentence_attention_mil": True,
            "attention_temperature": ATTENTION_TEMPERATURE,
            "sentence_budget_per_batch": SENTENCE_BUDGET,
            "maximum_bags_per_batch": MAX_BAGS_PER_BATCH,
            "weak_repeat_auxiliary_weight": REPEAT_AUX_WEIGHT,
            "bag_summary": bag_summary,
            "initialised_from": str(INITIAL_CHECKPOINT),
            "train_loss": train_loss,
        },
        "fixed_policy": {
            "replace_fraction_of_v3_prototype_component": REPLACEMENT_WEIGHT,
            "effective_total_weight": component_weight * REPLACEMENT_WEIGHT,
            "topk_ratio": TOPK_RATIO,
        },
        "baseline_macro_f1": baseline,
        "v17_teacher_candidate_macro_f1": v17,
        "definition_mil_standalone_macro_f1": standalone,
        "candidate_macro_f1": score,
        "delta": score - baseline,
        "per_label": per_label,
        "promising_for_full_oof": bool(score >= baseline + .005),
        "adopted": False,
    }
    RESULTS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    return payload


if __name__ == "__main__":
    train_fold0()
