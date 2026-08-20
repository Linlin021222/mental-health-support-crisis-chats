"""Leak-free sentence/token evidence extraction for Task 1 (V25).

The previous evidence system predicts only start/end points and then ranks a
large hand-built candidate pool.  V25 instead learns two aligned objectives:

* whether an offset-preserving clause contains evidence; and
* which *tokens* inside that clause belong to a gold evidence phrase.

Hyper-parameters and the epoch are selected on one inner, user-disjoint OOF
fold.  The chosen recipe is then refitted on every outer-training user and is
evaluated exactly once on the untouched outer holdout used by the other Task 1
experiments.  The stable risk predictions are deliberately left unchanged so
that this experiment isolates evidence quality.
"""
from __future__ import annotations

import copy
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup

from baseline import _post_phrase_f1
from configs.config import config
from inference.task1_evidence_v4 import load_evidence_calibration
from preprocess.preprocess import load_train_data
from trainer.task1_oof_stack_v20 import (
    INNER_FOLDS,
    OUTPUT as V20_OUTPUT,
    _baseline_evidence,
)
from utils.seed import seed_everything
from utils.task1_metric import task1_score


OUTPUT = config.OUTPUT_DIR / "task1_atomic_v25"
RESULTS = OUTPUT / "results.json"
CALIBRATION = OUTPUT / "calibration.json"
INNER_CHECKPOINT = OUTPUT / "inner_model.pt"
OUTER_CHECKPOINT = OUTPUT / "outer_model.pt"
TRAINING_VERSION = "task1-atomic-sentence-token-v25"
MAX_LENGTH = 160
BATCH_SIZE = 8
ACCUMULATION = 2
MAX_EPOCHS = 4
INNER_CALIBRATION_FOLD = 3

HARD_CUE = re.compile(
    r"\b(?:suicid\w*|kill(?:ing)?\s+myself|end\s+(?:my|this)\s+life|"
    r"want\s+to\s+die|wanna\s+die|wish\s+(?:i|to).*?die|overdos\w*|"
    r"hang(?:ing)?\s+myself|slit(?:ting)?\s+my|jump(?:ing)?\s+(?:off|from)|"
    r"shot\s+myself|attempt(?:ed|ing)?|self[- ]?harm)\b",
    re.IGNORECASE,
)


def _gold_spans(text: str, phrases: list[str]) -> list[tuple[int, int]]:
    """Locate annotations without normalising the source text or its offsets."""
    found: list[tuple[int, int]] = []
    for phrase in phrases:
        pieces = [piece for piece in re.split(r"\s+", str(phrase).strip()) if piece]
        if not pieces:
            continue
        pattern = r"\s+".join(re.escape(piece) for piece in pieces)
        matches = list(re.finditer(pattern, text, flags=re.IGNORECASE))
        if not matches:
            # Excel occasionally normalises whitespace next to punctuation.
            start = text.casefold().find(str(phrase).casefold())
            if start >= 0:
                matches = [type("Match", (), {"start": lambda self, s=start: s,
                                                "end": lambda self, e=start + len(str(phrase)): e})()]
        found.extend((int(match.start()), int(match.end())) for match in matches)
    return sorted(set(found))


def _trim_span(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _segment_spans(text: str) -> list[tuple[int, int]]:
    """Sentence/clause spans with overlapping windows for long Reddit lines."""
    coarse: list[tuple[int, int]] = []
    start = 0
    for match in re.finditer(r"(?:\r?\n+|[.!?;]+(?:[\"')\]]*)\s+)", text):
        end = match.end()
        span = _trim_span(text, start, end)
        if span[1] > span[0]:
            coarse.append(span)
        start = end
    span = _trim_span(text, start, len(text))
    if span[1] > span[0]:
        coarse.append(span)
    if not coarse and text.strip():
        coarse = [_trim_span(text, 0, len(text))]

    result: list[tuple[int, int]] = []
    for left, right in coarse:
        words = list(re.finditer(r"\S+", text[left:right]))
        if len(words) <= 70:
            result.append((left, right))
            continue
        # 70-word windows fit comfortably in 160 DeBERTa subwords for typical
        # Reddit text; the 15-word overlap protects phrases at a boundary.
        for word_start in range(0, len(words), 55):
            word_end = min(len(words), word_start + 70)
            window_left = left + words[word_start].start()
            window_right = left + words[word_end - 1].end()
            result.append(_trim_span(text, window_left, window_right))
            if word_end == len(words):
                break
    # Preserve order while removing identical windows.
    return list(dict.fromkeys(result))


def _overlaps(start: int, end: int, gold: list[tuple[int, int]]) -> bool:
    return any(start < gold_end and end > gold_start for gold_start, gold_end in gold)


class AtomicDataset(Dataset):
    def __init__(self, examples: list[dict]):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


def _collate(items: list[dict]) -> dict:
    tensor_keys = ("input_ids", "attention_mask", "token_mask", "token_labels", "offsets")
    batch = {key: torch.stack([item[key] for item in items]) for key in tensor_keys}
    batch["sentence_label"] = torch.tensor(
        [item["sentence_label"] for item in items], dtype=torch.float32
    )
    for key in ("post_index", "segment_start", "segment_end"):
        batch[key] = [item[key] for item in items]
    return batch


def _build_examples(frame, indices, tokenizer, training: bool) -> list[dict]:
    examples: list[dict] = []
    raw_segments: list[dict] = []
    for global_index in tqdm(
        list(map(int, indices)), desc=f"V25 segment posts (training={training})"
    ):
        row = frame.iloc[global_index]
        text = str(row.text)
        gold = _gold_spans(text, list(row.evidence))
        candidates = []
        for order, (segment_start, segment_end) in enumerate(_segment_spans(text)):
            positive = _overlaps(segment_start, segment_end, gold)
            hard = bool(HARD_CUE.search(text[segment_start:segment_end]))
            candidates.append((order, segment_start, segment_end, positive, hard))

        if training:
            positives = [item for item in candidates if item[3]]
            hard_negatives = [item for item in candidates if not item[3] and item[4]][:6]
            easy = [item for item in candidates if not item[3] and not item[4]]
            # Always retain the opening/title and two deterministic contextual
            # negatives.  This teaches quoted/third-person suicidal language
            # without letting long negative posts dominate the loss.
            selected = positives + hard_negatives
            selected_orders = {item[0] for item in selected}
            for item in easy:
                if item[0] == 0 or len([x for x in selected if not x[3] and not x[4]]) < 2:
                    if item[0] not in selected_orders:
                        selected.append(item); selected_orders.add(item[0])
            candidates = sorted(selected, key=lambda item: item[0])

        for _, segment_start, segment_end, _, _ in candidates:
            raw_segments.append({
                "text": text,
                "gold": gold,
                "segment": text[segment_start:segment_end],
                "post_index": global_index,
                "segment_start": int(segment_start),
                "segment_end": int(segment_end),
            })

    # A single batched Rust-tokenizer call is over an order of magnitude faster
    # than invoking the tokenizer separately for every sentence.
    print(f"V25 batch tokenizing {len(raw_segments)} segments", flush=True)
    encoded_all = tokenizer(
        [item["segment"] for item in raw_segments],
        max_length=MAX_LENGTH, truncation=True, padding="max_length",
        return_offsets_mapping=True,
    )
    for row_index, raw in enumerate(tqdm(raw_segments, desc="V25 align token labels")):
            offsets = torch.tensor(
                encoded_all["offset_mapping"][row_index], dtype=torch.long
            )
            attention = torch.tensor(
                encoded_all["attention_mask"][row_index], dtype=torch.long
            )
            token_mask = (
                attention.bool() & (offsets[:, 1] > offsets[:, 0])
            )
            labels = torch.zeros(MAX_LENGTH, dtype=torch.float32)
            for token_index, (local_start, local_end) in enumerate(offsets.tolist()):
                if local_end <= local_start:
                    continue
                absolute_start = raw["segment_start"] + local_start
                absolute_end = raw["segment_start"] + local_end
                labels[token_index] = float(
                    _overlaps(absolute_start, absolute_end, raw["gold"])
                )
            # If a segment touches a gold span only outside a tokenizer-truncated
            # tail, it is not a positive training instance for this encoding.
            sentence_label = float(bool((labels > 0).any()))
            examples.append({
                "input_ids": torch.tensor(
                    encoded_all["input_ids"][row_index], dtype=torch.long
                ),
                "attention_mask": attention,
                "token_mask": token_mask,
                "token_labels": labels,
                "offsets": offsets,
                "sentence_label": sentence_label,
                "post_index": raw["post_index"],
                "segment_start": raw["segment_start"],
                "segment_end": raw["segment_end"],
            })
    positives = int(sum(item["sentence_label"] for item in examples))
    print(
        f"V25 examples: posts={len(indices)} segments={len(examples)} "
        f"positive_segments={positives} training={training}", flush=True,
    )
    return examples


class AtomicEvidenceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(
            config.MODEL_NAME, local_files_only=True, dtype=torch.float32
        )
        self.encoder.gradient_checkpointing_disable()
        hidden = int(self.encoder.config.hidden_size)
        self.dropout = nn.Dropout(0.12)
        self.token_head = nn.Linear(hidden, 1)
        self.sentence_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Dropout(0.12),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, input_ids, attention_mask, token_mask):
        hidden = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        token_logits = self.token_head(self.dropout(hidden)).squeeze(-1)
        weights = token_mask.to(hidden.dtype).unsqueeze(-1)
        pooled = (hidden * weights).sum(1) / weights.sum(1).clamp_min(1.0)
        sentence_logits = self.sentence_head(self.dropout(pooled)).squeeze(-1)
        return token_logits, sentence_logits


def _loss_weights(examples: list[dict], device):
    token_positive = sum(float(item["token_labels"].sum()) for item in examples)
    token_total = sum(float(item["token_mask"].sum()) for item in examples)
    sentence_positive = sum(float(item["sentence_label"]) for item in examples)
    token_weight = min(15.0, math.sqrt(max(1.0, (token_total - token_positive) / max(1.0, token_positive))))
    sentence_weight = min(10.0, math.sqrt(max(1.0, (len(examples) - sentence_positive) / max(1.0, sentence_positive))))
    print(
        f"V25 loss weights: token={token_weight:.3f} sentence={sentence_weight:.3f}",
        flush=True,
    )
    return (
        torch.tensor(token_weight, device=device),
        torch.tensor(sentence_weight, device=device),
    )


def _batch_loss(model, batch, device, token_pos_weight, sentence_pos_weight):
    input_ids = batch["input_ids"].to(device)
    attention = batch["attention_mask"].to(device)
    token_mask = batch["token_mask"].to(device)
    token_labels = batch["token_labels"].to(device)
    sentence_labels = batch["sentence_label"].to(device)
    token_logits, sentence_logits = model(input_ids, attention, token_mask)
    raw_token = nn.functional.binary_cross_entropy_with_logits(
        token_logits, token_labels, pos_weight=token_pos_weight, reduction="none"
    )
    token_bce = (raw_token * token_mask).sum() / token_mask.sum().clamp_min(1)
    probabilities = torch.sigmoid(token_logits) * token_mask
    intersection = (probabilities * token_labels).sum(1)
    dice = 1.0 - ((2.0 * intersection + 1.0) /
                  (probabilities.sum(1) + token_labels.sum(1) + 1.0)).mean()
    sentence_bce = nn.functional.binary_cross_entropy_with_logits(
        sentence_logits, sentence_labels, pos_weight=sentence_pos_weight
    )
    return token_bce + 0.30 * dice + 0.45 * sentence_bce


def _make_optimizer(model):
    head = list(model.token_head.parameters()) + list(model.sentence_head.parameters())
    head_ids = {id(parameter) for parameter in head}
    backbone = [parameter for parameter in model.parameters() if id(parameter) not in head_ids]
    return AdamW([
        {"params": backbone, "lr": 8e-6},
        {"params": head, "lr": 4e-5},
    ], weight_decay=0.01)


def _train_epochs(model, examples, device, epochs, epoch_callback=None):
    loader = DataLoader(
        AtomicDataset(examples), batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, collate_fn=_collate, pin_memory=True,
    )
    optimizer = _make_optimizer(model)
    updates = math.ceil(len(loader) / ACCUMULATION) * epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=max(1, int(0.1 * updates)),
        num_training_steps=max(1, updates),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    token_weight, sentence_weight = _loss_weights(examples, device)
    history = []
    for epoch in range(1, epochs + 1):
        model.train(); optimizer.zero_grad(set_to_none=True); losses = []
        progress = tqdm(loader, desc=f"V25 epoch {epoch}/{epochs}")
        for step, batch in enumerate(progress, 1):
            with torch.autocast(device_type="cuda", enabled=True):
                loss = _batch_loss(
                    model, batch, device, token_weight, sentence_weight
                ) / ACCUMULATION
            scaler.scale(loss).backward()
            losses.append(float(loss.detach()) * ACCUMULATION)
            if step % ACCUMULATION == 0 or step == len(loader):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                old_scale = scaler.get_scale()
                scaler.step(optimizer); scaler.update()
                if scaler.get_scale() >= old_scale:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
        mean_loss = float(np.mean(losses))
        history.append({"epoch": epoch, "train_loss": mean_loss})
        print(f"V25 epoch={epoch} train_loss={mean_loss:.4f}", flush=True)
        if epoch_callback is not None:
            epoch_callback(epoch, model, history[-1])
    return history


@torch.no_grad()
def _infer(model, examples, device, description):
    loader = DataLoader(
        AtomicDataset(examples), batch_size=BATCH_SIZE * 2, shuffle=False,
        num_workers=0, collate_fn=_collate, pin_memory=True,
    )
    rows = []; model.eval(); cursor = 0
    for batch in tqdm(loader, desc=description):
        token_logits, sentence_logits = model(
            batch["input_ids"].to(device), batch["attention_mask"].to(device),
            batch["token_mask"].to(device),
        )
        token_probabilities = torch.sigmoid(token_logits).cpu().numpy()
        sentence_probabilities = torch.sigmoid(sentence_logits).cpu().numpy()
        for local in range(len(batch["post_index"])):
            example = examples[cursor]
            rows.append({
                "post_index": int(example["post_index"]),
                "segment_start": int(example["segment_start"]),
                "offsets": example["offsets"].numpy(),
                "token_mask": example["token_mask"].numpy(),
                "token_probability": token_probabilities[local],
                "sentence_probability": float(sentence_probabilities[local]),
            })
            cursor += 1
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["post_index"]].append(row)
    return grouped


def _normalise(value):
    return " ".join(str(value).casefold().split())


def _atomic_candidates(text, segments, threshold, sentence_threshold, max_tokens):
    candidates = []
    for segment in segments:
        sentence_probability = segment["sentence_probability"]
        if sentence_probability < sentence_threshold:
            continue
        probability = segment["token_probability"] * math.sqrt(sentence_probability)
        offsets = segment["offsets"]
        valid = segment["token_mask"].astype(bool)
        active = np.flatnonzero(valid & (probability >= threshold))
        if not len(active):
            continue
        runs = []
        run = [int(active[0])]
        for token_index in active[1:]:
            token_index = int(token_index)
            previous = run[-1]
            gap = int(offsets[token_index, 0] - offsets[previous, 1])
            if token_index == previous + 1 and gap <= 3:
                run.append(token_index)
            else:
                runs.append(run); run = [token_index]
        runs.append(run)
        for run in runs:
            if len(run) > max_tokens:
                best_start = max(
                    range(0, len(run) - max_tokens + 1),
                    key=lambda start: float(probability[run[start:start + max_tokens]].mean()),
                )
                run = run[best_start:best_start + max_tokens]
            absolute_start = segment["segment_start"] + int(offsets[run[0], 0])
            absolute_end = segment["segment_start"] + int(offsets[run[-1], 1])
            absolute_start, absolute_end = _trim_span(text, absolute_start, absolute_end)
            phrase = text[absolute_start:absolute_end]
            if phrase:
                score = float(probability[run].mean())
                candidates.append((score, absolute_start, absolute_end, phrase))
    # Overlapping segmentation creates duplicate spans.  Prefer the higher
    # confidence boundary and keep non-overlapping evidence mentions.
    selected = []
    for candidate in sorted(candidates, key=lambda item: (-item[0], len(item[3]))):
        _, start, end, phrase = candidate
        normal = _normalise(phrase)
        duplicate = False
        for _, old_start, old_end, old_phrase in selected:
            old_normal = _normalise(old_phrase)
            if normal == old_normal or (
                max(start, old_start) < min(end, old_end)
                and (normal in old_normal or old_normal in normal)
            ):
                duplicate = True; break
        if not duplicate:
            selected.append(candidate)
    return selected


def _fuse(atomic, baseline, mode, topk, gate):
    if mode == "atomic":
        values = list(atomic)
    elif mode == "gated_replace":
        values = list(atomic) if atomic and atomic[0][0] >= gate else [
            (0.0, -1, -1, phrase) for phrase in baseline
        ]
    elif mode == "atomic_then_baseline":
        values = list(atomic) + [(0.0, -1, -1, phrase) for phrase in baseline]
    else:
        values = [(0.0, -1, -1, phrase) for phrase in baseline]
    result = []
    for _, _, _, phrase in values:
        normal = _normalise(phrase)
        if not normal or any(normal == _normalise(old) or normal in _normalise(old)
                             or _normalise(old) in normal for old in result):
            continue
        result.append(phrase)
        if len(result) == topk:
            break
    return result


def _predict_evidence(frame, indices, grouped, risks, baselines, parameters):
    predictions = []; scores = []
    for global_index in map(int, indices):
        if int(risks[global_index]) == config.RISK_LABELS["Indicator"]:
            predictions.append([]); scores.append(0.0); continue
        text = str(frame.iloc[global_index].text)
        atomic = _atomic_candidates(
            text, grouped.get(global_index, []),
            parameters["threshold"], parameters["sentence_threshold"],
            parameters["max_tokens"],
        )
        evidence = _fuse(
            atomic, baselines[global_index], parameters["mode"],
            parameters["topk"], parameters["gate"],
        )
        predictions.append(evidence)
        scores.append(float(atomic[0][0]) if atomic else 0.0)
    return predictions, scores


def _grid_search(frame, indices, grouped, risks, baselines):
    gold = [list(frame.iloc[int(index)].evidence) for index in indices]
    rows = []
    for threshold in (0.25, 0.35, 0.45, 0.55, 0.65):
        for sentence_threshold in (0.20, 0.40, 0.60):
            for max_tokens in (4, 8, 12):
                for topk in (1, 2, 3, 4):
                    for mode in ("atomic", "gated_replace", "atomic_then_baseline"):
                        parameters = {
                            "threshold": threshold,
                            "sentence_threshold": sentence_threshold,
                            "max_tokens": max_tokens,
                            "topk": topk,
                            "mode": mode,
                            "gate": 0.65,
                        }
                        predictions, _ = _predict_evidence(
                            frame, indices, grouped, risks, baselines, parameters
                        )
                        phrase = float(np.mean([
                            _post_phrase_f1(prediction, target)
                            for prediction, target in zip(predictions, gold)
                        ]))
                        rows.append({**parameters, "phrase_f1": phrase})
    rows.sort(key=lambda row: row["phrase_f1"], reverse=True)
    return rows


def _load_records():
    inner_records = []; membership = {}
    for fold in range(INNER_FOLDS):
        saved = torch.load(
            V20_OUTPUT / f"inner_fold{fold}_raw.pt",
            map_location="cpu", weights_only=False,
        )
        for record in saved["records"]:
            index = int(record["global_index"])
            inner_records.append(record); membership[index] = fold
    inner_records.sort(key=lambda row: int(row["global_index"]))
    outer_raw = torch.load(
        config.OUTPUT_DIR / "task1_evidence_v4" / "fold0_raw.pt",
        map_location="cpu", weights_only=False,
    )
    return inner_records, membership, outer_raw


def _bootstrap(groups, baseline, candidate):
    unique = np.unique(groups); rng = np.random.default_rng(config.SEED + 2525)
    values = []
    for _ in range(4000):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        positions = np.concatenate([np.flatnonzero(groups == user) for user in sampled])
        values.append(float(candidate[positions].mean() - baseline[positions].mean()))
    values = np.asarray(values)
    return {
        "mean_phrase_delta": float(values.mean()),
        "p05_phrase_delta": float(np.quantile(values, 0.05)),
        "p95_phrase_delta": float(np.quantile(values, 0.95)),
        "positive_fraction": float((values > 0).mean()),
    }


def train_task1_atomic_v25():
    if not torch.cuda.is_available():
        raise RuntimeError("Task 1 atomic V25 requires CUDA")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    seed_everything(config.SEED + 2525)
    device = torch.device("cuda")
    frame = load_train_data().reset_index(drop=True)
    labels = frame.risk_label.to_numpy()
    groups = frame.anon_user_id.astype(str).to_numpy()
    outer_train, outer_valid = next(StratifiedGroupKFold(
        n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED
    ).split(np.zeros(len(frame)), labels, groups))
    inner_records, membership, outer_raw = _load_records()
    if not np.array_equal(np.asarray(outer_raw["valid_idx"]), outer_valid):
        raise ValueError("V25 outer holdout differs from the stable V4 holdout")
    inner_fit = np.asarray([
        index for index in outer_train if membership[int(index)] != INNER_CALIBRATION_FOLD
    ], dtype=np.int64)
    inner_cal = np.asarray([
        index for index in outer_train if membership[int(index)] == INNER_CALIBRATION_FOLD
    ], dtype=np.int64)
    if set(groups[inner_fit]) & set(groups[inner_cal]):
        raise ValueError("V25 inner fit/calibration users overlap")

    calibration = load_evidence_calibration()
    print("V25 loaded clean user-disjoint splits and stable V4 calibration", flush=True)
    inner_by_index = {int(row["global_index"]): row for row in inner_records}
    outer_records = outer_raw["records"]
    outer_by_index = {
        int(index): record for index, record in zip(outer_valid, outer_records)
    }
    inner_risks = {
        int(index): int(inner_by_index[int(index)]["risk"])
        for index in inner_cal
    }
    outer_risks = {index: int(record["risk"]) for index, record in outer_by_index.items()}
    print(f"V25 decoding {len(inner_cal)} inner OOF baselines", flush=True)
    inner_baselines = {
        int(index): _baseline_evidence(inner_by_index[int(index)], calibration)
        for index in inner_cal
    }
    print(f"V25 decoding {len(outer_valid)} untouched outer baselines", flush=True)
    outer_baselines = {
        index: _baseline_evidence(record, calibration)
        for index, record in outer_by_index.items()
    }
    print("V25 baseline decoding complete", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        config.MODEL_NAME, use_fast=True, local_files_only=True
    )

    fit_examples = _build_examples(frame, inner_fit, tokenizer, training=True)
    cal_examples = _build_examples(frame, inner_cal, tokenizer, training=False)
    model = AtomicEvidenceModel().to(device)
    selected = {"phrase_f1": -1.0}; selected_state = None; grid_top = []

    def calibrate_epoch(epoch, current_model, history_row):
        nonlocal selected, selected_state, grid_top
        grouped = _infer(current_model, cal_examples, device, f"V25 inner cal epoch {epoch}")
        grid = _grid_search(
            frame, inner_cal, grouped, inner_risks, inner_baselines
        )
        history_row["calibration_phrase_f1"] = grid[0]["phrase_f1"]
        print(
            f"V25 epoch={epoch} inner_phrase_f1={grid[0]['phrase_f1']:.6f} "
            f"params={{{', '.join(f'{k}: {v}' for k, v in grid[0].items() if k != 'phrase_f1')}}}",
            flush=True,
        )
        if grid[0]["phrase_f1"] > selected["phrase_f1"]:
            selected = {"epoch": epoch, **grid[0]}
            selected_state = copy.deepcopy({
                key: value.detach().cpu() for key, value in current_model.state_dict().items()
            })
            grid_top = grid[:20]

    history = _train_epochs(
        model, fit_examples, device, MAX_EPOCHS, epoch_callback=calibrate_epoch
    )
    torch.save({
        "training_version": TRAINING_VERSION, "state_dict": selected_state,
        "selected": selected, "history": history,
    }, INNER_CHECKPOINT)
    del model, fit_examples, cal_examples, selected_state
    torch.cuda.empty_cache()

    # Refit the selected epoch count on all outer-training users.  No outer
    # labels are inspected during this stage.
    seed_everything(config.SEED + 2526)
    full_examples = _build_examples(frame, outer_train, tokenizer, training=True)
    final_model = AtomicEvidenceModel().to(device)
    refit_history = _train_epochs(
        final_model, full_examples, device, int(selected["epoch"])
    )
    torch.save({
        "training_version": TRAINING_VERSION,
        "state_dict": final_model.state_dict(), "selected": selected,
        "history": refit_history,
    }, OUTER_CHECKPOINT)
    outer_examples = _build_examples(frame, outer_valid, tokenizer, training=False)
    grouped = _infer(final_model, outer_examples, device, "V25 untouched outer holdout")
    predictions, _ = _predict_evidence(
        frame, outer_valid, grouped, outer_risks, outer_baselines, selected
    )
    baseline_values = np.asarray([
        _post_phrase_f1(outer_baselines[int(index)], list(frame.iloc[int(index)].evidence))
        for index in outer_valid
    ], dtype=np.float32)
    candidate_values = np.asarray([
        _post_phrase_f1(prediction, list(frame.iloc[int(index)].evidence))
        for prediction, index in zip(predictions, outer_valid)
    ], dtype=np.float32)
    truth = labels[outer_valid]
    risk = np.asarray([outer_risks[int(index)] for index in outer_valid])
    risk_f1 = float(f1_score(truth, risk, average="weighted", zero_division=0))
    baseline_task = task1_score(risk_f1, float(baseline_values.mean()))
    candidate_task = task1_score(risk_f1, float(candidate_values.mean()))
    bootstrap = _bootstrap(groups[outer_valid], baseline_values, candidate_values)
    adopted = bool(
        candidate_task >= baseline_task + 0.005
        and bootstrap["positive_fraction"] >= 0.80
    )
    payload = {
        "training_version": TRAINING_VERSION,
        "evaluation_scope": "inner user-disjoint calibration; refit outer train; one untouched outer evaluation",
        "split": {
            "inner_fit_posts": int(len(inner_fit)),
            "inner_calibration_posts": int(len(inner_cal)),
            "outer_refit_posts": int(len(outer_train)),
            "outer_holdout_posts": int(len(outer_valid)),
            "user_overlap": 0,
        },
        "selected": selected,
        "inner_history": history,
        "refit_history": refit_history,
        "calibration_top20": grid_top,
        "baseline": {
            "risk_f1": risk_f1,
            "phrase_f1": float(baseline_values.mean()),
            "task1": baseline_task,
        },
        "candidate": {
            "risk_f1": risk_f1,
            "phrase_f1": float(candidate_values.mean()),
            "task1": candidate_task,
            "improved_posts": int((candidate_values > baseline_values).sum()),
            "worsened_posts": int((candidate_values < baseline_values).sum()),
        },
        "user_cluster_bootstrap": bootstrap,
        "adopted": adopted,
    }
    RESULTS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    CALIBRATION.write_text(json.dumps({
        "training_version": TRAINING_VERSION,
        "adopted": adopted,
        "strict_task1": candidate_task,
        "bootstrap_positive_fraction": bootstrap["positive_fraction"],
        **{key: selected[key] for key in (
            "epoch", "threshold", "sentence_threshold", "max_tokens",
            "topk", "mode", "gate",
        )},
    }, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    return payload


if __name__ == "__main__":
    train_task1_atomic_v25()
