"""Sentence-level weak factor evidence on a strict user-disjoint fold.

Repeated labels in the released workbook are not duplicate output classes.
They are used here as a weak indication that several distinct passages may
support the same binary factor.  A boundary-aware V15 model mines at most
three sentences per positive post/label pair from the outer training users.
The resulting evidence is auditable and is then used to continue the NLI
cross encoder with sentence-level positive and label-confusion negatives.
"""
from __future__ import annotations

from collections import defaultdict
import json
import random
import re

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
from utils.seed import seed_everything


OUTPUT = config.OUTPUT_DIR / "factor_sentence_evidence_v16"
RESULTS = OUTPUT / "fold0_results.json"
PSEUDO_EVIDENCE = OUTPUT / "fold0_pseudo_evidence.jsonl"
CHECKPOINT = OUTPUT / "fold0_model.pt"
VALID_PREDICTIONS = OUTPUT / "fold0_valid.npz"
MINER_CHECKPOINT = config.OUTPUT_DIR / "factor_semantic_contrast_v15" / "fold0_model.pt"
ACCEPTED_CROSS = config.OUTPUT_DIR / "factor_cross_encoder_v2" / "oof_predictions.npz"
TRAINING_VERSION = "sentence-factor-evidence-v16"

# These values are fixed before looking at V16 validation labels.
PROMPT_INDICES = (0, 4)  # formal definition + full boundary-aware description
REPLACEMENT_WEIGHT = 0.20
TOPK_RATIO = 1.10
MAX_SENTENCES = 14
MAX_EVIDENCE_SENTENCES = 3
MAX_LENGTH = 192
INFER_BATCH = 32
TRAIN_BATCH = 12
ACCUMULATION = 2
LEARNING_RATE = 2e-6
# V16 intentionally used zero to measure the unfiltered weak-supervision
# baseline.  Follow-up experiments may raise this for semi-automatic review.
MIN_EVIDENCE_SCORE = 0.0


def _sentences(text: str) -> list[str]:
    """Split Reddit text while retaining useful short/line-delimited clauses."""
    text = str(text).replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return [""]
    pieces = re.split(r"\n+|(?<=[.!?])\s+", text)
    result = []
    for piece in pieces:
        piece = re.sub(r"\s+", " ", piece).strip()
        if not piece:
            continue
        words = piece.split()
        # Posts often contain long punctuation-free paragraphs.  Local windows
        # preserve evidence specificity better than calling the whole post a sentence.
        if len(words) <= 72:
            result.append(piece)
        else:
            for start in range(0, len(words), 60):
                window = " ".join(words[start:start + 72]).strip()
                if window:
                    result.append(window)
                if start + 72 >= len(words):
                    break
    # Remove duplicated lines such as quoted/reposted text.
    unique, seen = [], set()
    for sentence in result:
        key = re.sub(r"\W+", " ", sentence).casefold().strip()
        if key and key not in seen:
            seen.add(key); unique.append(sentence)
    if not unique:
        return [text]
    if len(unique) <= MAX_SENTENCES:
        return unique
    positions = np.linspace(0, len(unique) - 1, MAX_SENTENCES).round().astype(int)
    return [unique[int(i)] for i in np.unique(positions)]


def _load_model(checkpoint, device):
    model = AutoModelForSequenceClassification.from_pretrained(
        config.FACTOR_NLI_MODEL_NAME, dtype=torch.float32, local_files_only=True,
    )
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    return model.to(device)


@torch.no_grad()
def _score_pairs(model, tokenizer, premises, hypotheses, device, desc):
    """Return binary entailment probabilities for aligned text/prompt pairs."""
    model.eval(); entailment = _entailment_index(model)
    output = np.empty(len(premises), dtype=np.float32)
    for start in tqdm(range(0, len(premises), INFER_BATCH), desc=desc):
        stop = min(start + INFER_BATCH, len(premises))
        encoded = tokenizer(
            premises[start:stop], hypotheses[start:stop], padding=True,
            truncation="only_first", max_length=MAX_LENGTH, return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(**encoded).logits.float()
        if logits.shape[-1] == 1:
            scores = torch.sigmoid(logits[:, 0])
        else:
            scores = torch.softmax(logits, dim=-1)[:, entailment]
        output[start:stop] = scores.cpu().numpy()
    return output


def _mine_evidence(frame, train_idx, targets, counts, model, tokenizer, device):
    """Mine only outer-train positives; validation labels never enter this step."""
    candidate_meta, premises, hypotheses = [], [], []
    sentence_cache = {}
    for global_row in tqdm(train_idx, desc="build sentence candidates"):
        row_sentences = _sentences(frame.text.iloc[global_row])
        sentence_cache[int(global_row)] = row_sentences
        for label in np.flatnonzero(targets[global_row]):
            for sentence_index, sentence in enumerate(row_sentences):
                for prompt_index in PROMPT_INDICES:
                    candidate_meta.append((int(global_row), int(label), sentence_index,
                                           int(prompt_index)))
                    premises.append(sentence)
                    hypotheses.append(FACTOR_CLASSIFIER_PROMPTS[label][prompt_index])
    scores = _score_pairs(model, tokenizer, premises, hypotheses, device,
                          "mine positive evidence")
    grouped = defaultdict(lambda: defaultdict(list))
    for meta, score in zip(candidate_meta, scores):
        row, label, sentence_index, prompt_index = meta
        grouped[(row, label)][sentence_index].append(float(score))

    records = []
    supports = targets[train_idx].sum(0).clip(min=1)
    max_support = float(supports.max())
    train_pairs = []
    rng = random.Random(config.SEED + 160)
    for (row, label), per_sentence in grouped.items():
        ranked = sorted(
            ((float(np.mean(values)), sentence_index)
             for sentence_index, values in per_sentence.items()),
            reverse=True,
        )
        requested = min(MAX_EVIDENCE_SENTENCES,
                        max(1, int(round(float(counts[row, label])))))
        selected = [item for item in ranked if item[0] >= MIN_EVIDENCE_SCORE]
        selected = selected[:min(requested, len(selected))]
        selected_rows = []
        absent_confusions = []
        for group in CONFUSION_GROUPS:
            if label in group:
                absent_confusions.extend(x for x in group if not targets[row, x])
        absent_confusions = sorted(set(absent_confusions))
        tail_weight = min(4.0, float((max_support / supports[label]) ** .30))
        for rank, (confidence, sentence_index) in enumerate(selected):
            sentence = sentence_cache[row][sentence_index]
            # Keep low-confidence tail evidence but let high-confidence pairs
            # contribute more.  The label itself remains strictly train-only.
            confidence_weight = .65 + .70 * float(np.clip(confidence, 0., 1.))
            positive_weight = tail_weight * confidence_weight
            for prompt_index in PROMPT_INDICES:
                train_pairs.append((sentence, label, prompt_index, 1, positive_weight))
            negative_label = None
            if absent_confusions:
                negative_label = absent_confusions[(rank + row + label) % len(absent_confusions)]
                train_pairs.append((sentence, negative_label, 4, 0, 1.20))
            selected_rows.append({
                "sentence": sentence, "sentence_index": int(sentence_index),
                "semantic_score": confidence, "hard_negative_label": (
                    None if negative_label is None else config.FACTOR_LABELS[negative_label]
                ),
            })
        # One post-level random absent label prevents a narrow model that only
        # understands the hand-curated confusion graph.
        absent = np.flatnonzero(~targets[row].astype(bool)).tolist()
        if selected and absent:
            random_label = rng.choice(absent)
            train_pairs.append((sentence_cache[row][selected[0][1]], random_label,
                                rng.choice(PROMPT_INDICES), 0, 1.0))
        records.append({
            "row_index": int(row), "row_id": str(frame.row_id.iloc[row]),
            "factor": config.FACTOR_LABELS[label],
            "annotation_count": int(counts[row, label]),
            "requested_sentence_count": int(requested),
            "selected_sentence_count": len(selected_rows),
            "selected": selected_rows,
        })
    rng.shuffle(train_pairs)
    return records, train_pairs


class _PairDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        return self.pairs[index]


def _collator(tokenizer):
    def collate(rows):
        sentences, labels, prompts, targets, weights = zip(*rows)
        hypotheses = [FACTOR_CLASSIFIER_PROMPTS[int(label)][int(prompt)]
                      for label, prompt in zip(labels, prompts)]
        encoded = tokenizer(
            list(sentences), hypotheses, padding=True, truncation="only_first",
            max_length=MAX_LENGTH, return_tensors="pt",
        )
        encoded["targets"] = torch.tensor(targets, dtype=torch.float32)
        encoded["weights"] = torch.tensor(weights, dtype=torch.float32)
        return encoded
    return collate


def _train(model, tokenizer, pairs, device):
    seed_everything(config.SEED + 160)
    loader = DataLoader(
        _PairDataset(pairs), batch_size=TRAIN_BATCH, shuffle=True,
        collate_fn=_collator(tokenizer), num_workers=0,
    )
    entailment = _entailment_index(model); not_entailment = 1 - entailment
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE,
                      weight_decay=config.WEIGHT_DECAY)
    updates = int(np.ceil(len(loader) / ACCUMULATION))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, max(1, int(updates * .08)), max(1, updates),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    model.train(); optimizer.zero_grad(set_to_none=True); losses = []
    for step, batch in enumerate(tqdm(loader, desc="sentence evidence training"), 1):
        targets = batch.pop("targets").to(device)
        weights = batch.pop("weights").to(device)
        inputs = {key: value.to(device) for key, value in batch.items()}
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(**inputs).logits
            margin = logits[:, entailment] - logits[:, not_entailment]
            raw = torch.nn.functional.binary_cross_entropy_with_logits(
                margin, targets, reduction="none",
            )
            loss = (raw * weights).mean() / ACCUMULATION
        scaler.scale(loss).backward(); losses.append(float(loss.detach()) * ACCUMULATION)
        if step % ACCUMULATION == 0 or step == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
            old_scale = scaler.get_scale(); scaler.step(optimizer); scaler.update()
            if scaler.get_scale() >= old_scale:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)
    return float(np.mean(losses))


def _predict_posts(model, tokenizer, texts, device):
    """Max-over-sentences MIL, averaged across two semantic descriptions."""
    meta, premises, hypotheses = [], [], []
    for row, text in enumerate(tqdm(texts, desc="build validation sentence pairs")):
        for sentence_index, sentence in enumerate(_sentences(text)):
            for label in range(config.NUM_FACTORS):
                for prompt_index in PROMPT_INDICES:
                    meta.append((row, sentence_index, label, prompt_index))
                    premises.append(sentence)
                    hypotheses.append(FACTOR_CLASSIFIER_PROMPTS[label][prompt_index])
    scores = _score_pairs(model, tokenizer, premises, hypotheses, device,
                          "sentence MIL inference")
    grouped = defaultdict(lambda: defaultdict(list))
    for (row, sentence_index, label, _), score in zip(meta, scores):
        grouped[(row, label)][sentence_index].append(float(score))
    output = np.zeros((len(texts), config.NUM_FACTORS), dtype=np.float32)
    for (row, label), sentence_scores in grouped.items():
        output[row, label] = max(float(np.mean(values))
                                 for values in sentence_scores.values())
    return output


def train_fold0():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if not MINER_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Missing V15 checkpoint: {MINER_CHECKPOINT}. Run "
            "`python main.py --mode factor-semantic-contrast-v15-fold0` first."
        )
    frame = load_train_data().reset_index(drop=True)
    targets = np.vstack(frame.factor_vector.to_numpy()).astype(np.int8)
    counts = np.vstack(frame.factor_counts.to_numpy()).astype(np.float32)
    fit_idx, valid_idx = next(StratifiedGroupKFold(
        n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED,
    ).split(np.zeros(len(frame)), frame.risk_label.to_numpy(),
            frame.anon_user_id.astype(str).to_numpy()))
    tokenizer = AutoTokenizer.from_pretrained(
        config.FACTOR_NLI_MODEL_NAME, use_fast=True, local_files_only=True,
    )
    device = torch.device(config.DEVICE)

    if CHECKPOINT.exists() and VALID_PREDICTIONS.exists() and PSEUDO_EVIDENCE.exists():
        saved = np.load(VALID_PREDICTIONS)
        if (str(saved["training_version"]) == TRAINING_VERSION
                and np.array_equal(saved["valid_indices"], valid_idx)):
            probability = saved["probabilities"].astype(np.float32)
            train_loss = float(saved["train_loss"])
            evidence_records = [json.loads(line) for line in
                                PSEUDO_EVIDENCE.read_text(encoding="utf-8").splitlines()
                                if line.strip()]
            pair_count = int(saved["pair_count"])
            print("V16 fold 0: resumed cached model and predictions", flush=True)
        else:
            raise RuntimeError("Existing V16 cache does not match the strict fold split.")
    else:
        model = _load_model(MINER_CHECKPOINT, device)
        evidence_records, pairs = _mine_evidence(
            frame, fit_idx, targets, counts, model, tokenizer, device,
        )
        PSEUDO_EVIDENCE.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in evidence_records) + "\n",
            encoding="utf-8",
        )
        pair_count = len(pairs)
        print(f"Mined {len(evidence_records)} positive post-label records; "
              f"training pairs={pair_count}", flush=True)
        train_loss = _train(model, tokenizer, pairs, device)
        probability = _predict_posts(
            model, tokenizer, frame.text.iloc[valid_idx].astype(str).tolist(), device,
        )
        torch.save(model.state_dict(), CHECKPOINT)
        np.savez_compressed(
            VALID_PREDICTIONS, probabilities=probability, valid_indices=valid_idx,
            train_loss=train_loss, pair_count=pair_count,
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
    prevalence = targets[fit_idx].mean(0)
    truth = targets[valid_idx]
    baseline_pred = _rank_decode(current[valid_idx], prevalence, TOPK_RATIO)
    candidate_pred = _rank_decode(candidate, prevalence, TOPK_RATIO)
    standalone_pred = _rank_decode(probability, prevalence, TOPK_RATIO)
    baseline = float(f1_score(truth, baseline_pred, average="macro", zero_division=0))
    score = float(f1_score(truth, candidate_pred, average="macro", zero_division=0))
    standalone = float(f1_score(truth, standalone_pred,
                                average="macro", zero_division=0))
    selected_counts = np.asarray([row["selected_sentence_count"]
                                  for row in evidence_records], dtype=np.int64)
    repeated_records = [row for row in evidence_records if row["annotation_count"] > 1]
    per_label = []
    for label, name in enumerate(config.FACTOR_LABELS):
        per_label.append({
            "label": name, "support": int(truth[:, label].sum()),
            "baseline_f1": float(f1_score(truth[:, label], baseline_pred[:, label],
                                           zero_division=0)),
            "candidate_f1": float(f1_score(truth[:, label], candidate_pred[:, label],
                                            zero_division=0)),
        })
    payload = {
        "training_version": TRAINING_VERSION,
        "strict_split": {"train_rows": len(fit_idx), "valid_rows": len(valid_idx),
                         "validation_users_excluded_from_mining": True},
        "sentence_evidence": {
            "positive_post_label_records": len(evidence_records),
            "records_with_repeated_annotation": len(repeated_records),
            "selected_sentences": int(selected_counts.sum()),
            "one_sentence_records": int((selected_counts == 1).sum()),
            "two_sentence_records": int((selected_counts == 2).sum()),
            "three_sentence_records": int((selected_counts == 3).sum()),
            "maximum_per_post_label": MAX_EVIDENCE_SENTENCES,
            "minimum_semantic_score": MIN_EVIDENCE_SCORE,
            "zero_sentence_records": int((selected_counts == 0).sum()),
            "audit_file": str(PSEUDO_EVIDENCE),
        },
        "training": {"pair_count": pair_count, "train_loss": train_loss,
                     "initialised_from": str(MINER_CHECKPOINT)},
        "fixed_policy": {
            "prompt_indices": list(PROMPT_INDICES),
            "replace_fraction_of_v3_prototype_component": REPLACEMENT_WEIGHT,
            "effective_total_weight": component_weight * REPLACEMENT_WEIGHT,
            "topk_ratio": TOPK_RATIO,
            "minimum_semantic_score": MIN_EVIDENCE_SCORE,
        },
        "baseline_macro_f1": baseline,
        "sentence_model_standalone_macro_f1": standalone,
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
