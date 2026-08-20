"""Qwen3-Reranker QLoRA V3 for the 24 suicide-related factors.

Each training item is a (factor hypothesis, Reddit post) pair with a strictly
constrained yes/no target.  This avoids placing a 24-way classifier on causal
token states that cannot all see the post.  The script preserves the official
19-risk/5-protective split in prompts, balanced sampling, diagnostics and
nested label-wise fusion, while allowing both groups to coexist in a post.
"""
from __future__ import annotations

import argparse
import ast
import gc
import json
import math
import os
from pathlib import Path
import random
import re

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)

from factor_taxonomy_v3 import (
    FACTORS,
    FACTOR_TO_ID,
    PROTECTIVE_INDICES,
    RISK_COUNT,
    RISK_INDICES,
    SPECS,
    factor_group,
    factor_query,
)


ROOT = Path(__file__).resolve().parent
WORK = Path(os.environ.get("KAGGLE_WORKING_DIR", "/kaggle/working"))
if not WORK.exists():
    WORK = ROOT / "working"
WORK.mkdir(parents=True, exist_ok=True)

SEED = 31415
N_FOLDS = 5
TRAINING_VERSION = "factor-pairwise-risk-protective-reranker-v3"
MODEL_CHOICES = {
    "qwen3-reranker-4b": "Qwen/Qwen3-Reranker-4B",
    "qwen3-reranker-8b": "Qwen/Qwen3-Reranker-8B",
}

PRESET = {
    "max_length": 1280,
    "rank": 16,
    "batch_size": 2,
    "accumulation": 8,
    "learning_rate": 2.0e-5,
    "weight_decay": 0.06,
    "warmup_ratio": 0.06,
    "pairs_per_epoch": 1920,
    # Macro F1 gives every factor equal weight, but the five protective labels
    # are semantically subtle.  A modest oversample is safer than forcing a
    # 50/50 risk/protective split, which would over-weight each protective label.
    "protective_share": 0.30,
    "hard_negative_share": 0.70,
    "ranking_weight": 0.25,
    "ranking_margin": 0.20,
}

SYSTEM_PREFIX = (
    "<|im_start|>system\n"
    "Judge whether the Document meets the factor requirements based on the "
    "Query and the Instruct. The answer can only be \"yes\" or \"no\"."
    "<|im_end|>\n<|im_start|>user\n"
)
INSTRUCTION = (
    "Determine whether the CURRENT Reddit post contains evidence that its "
    "AUTHOR experiences the specified suicide-related factor. Use the formal "
    "meaning and annotation boundaries. A risk factor and a protective factor "
    "may coexist. Do not infer diagnoses, events, support, access, purpose, or "
    "intent that the author did not express."
)
DECISION_SUFFIX = (
    "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
)

GENERAL_CUES = re.compile(
    r"depress|anxi|panic|ptsd|psych|bipolar|ocd|therap|medicat|pain|sick|"
    r"hospital|sleep|alcohol|drunk|drug|weed|addict|withdraw|hopeless|"
    r"no future|pointless|worthless|hate myself|burden|failure|angry|rage|"
    r"cry|emotion|grade|school|college|exam|job|money|debt|rent|homeless|"
    r"abuse|assault|rape|bull|hit me|threat|breakup|broke up|reject|alone|"
    r"lonely|nobody|no one|family|mother|father|parent|suicid|kill myself|"
    r"self harm|cutting|attempt|friend|brother|sister|died|death|trauma|"
    r"concentrat|confus|focus|memory|pills|gun|rope|bridge|knife|gay|lesbian|"
    r"bisexual|trans|queer|support|helped|listened|care|cope|coping|exercise|"
    r"hobby|hope|resilien|optimis|responsib|children|kid|pet|purpose|meaning|"
    r"reason to live|goal", re.I,
)


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def locate(name: str) -> Path:
    direct = ROOT / name
    if direct.exists():
        return direct
    if Path("/kaggle/input").exists():
        candidates = list(Path("/kaggle/input").rglob(name))
        if len(candidates) == 1:
            return candidates[0]
        raise FileNotFoundError(f"Could not uniquely locate {name}: {candidates}")
    raise FileNotFoundError(f"Missing package file: {direct}")


def parse_factor_cell(value) -> tuple[np.ndarray, np.ndarray]:
    if pd.isna(value) or not str(value).strip():
        values = []
    elif isinstance(value, (list, tuple)):
        values = list(value)
    else:
        values = ast.literal_eval(str(value))
    target = np.zeros(len(FACTORS), dtype=np.float32)
    counts = np.zeros(len(FACTORS), dtype=np.float32)
    for raw in values:
        key = str(raw).strip().casefold()
        if key not in FACTOR_TO_ID:
            raise ValueError(f"Unknown factor label: {raw!r}")
        index = FACTOR_TO_ID[key]
        target[index] = 1.0
        counts[index] += 1.0
    return target, counts


def load_train() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    frame = pd.read_excel(locate("train.xlsx"))
    required = {"row_id", "anon_user_id", "post", "factors"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"train.xlsx is missing columns: {sorted(missing)}")
    frame["post"] = frame["post"].fillna("").astype(str)
    frame["anon_user_id"] = frame["anon_user_id"].fillna(frame["row_id"]).astype(str)
    parsed = [parse_factor_cell(value) for value in frame["factors"]]
    targets = np.vstack([item[0] for item in parsed])
    counts = np.vstack([item[1] for item in parsed])
    return frame, targets, counts


def load_test() -> pd.DataFrame:
    frame = pd.read_excel(locate("leaderboard.xlsx"))
    if not {"row_id", "post"}.issubset(frame.columns):
        raise ValueError("leaderboard.xlsx must contain row_id and post")
    frame["post"] = frame["post"].fillna("").astype(str)
    return frame


def _distribute_even(total: int, labels: tuple[int, ...]) -> dict[int, int]:
    """Allocate an even number of examples to every label."""
    total = max(2 * len(labels), int(total))
    total -= total % 2
    blocks = total // 2
    base, remainder = divmod(blocks, len(labels))
    return {
        label: 2 * (base + (position < remainder))
        for position, label in enumerate(labels)
    }


def make_training_pairs(
    indices: np.ndarray,
    targets: np.ndarray,
    counts: np.ndarray,
    pairs_per_epoch: int,
    protective_share: float,
    hard_negative_share: float,
    seed: int,
) -> list[tuple[int, int, int]]:
    """Create matched positive/negative blocks for label-balanced ranking.

    Each adjacent pair has the same factor label and contains one positive and
    one negative post.  This guarantees that the ranking term is active even
    with a memory-safe batch size of two.
    """
    rng = np.random.default_rng(seed)
    protective_total = int(round(pairs_per_epoch * protective_share))
    protective_total -= protective_total % 2
    risk_total = pairs_per_epoch - protective_total
    risk_total -= risk_total % 2
    allocation = _distribute_even(risk_total, RISK_INDICES)
    allocation.update(_distribute_even(protective_total, PROTECTIVE_INDICES))

    indices = np.asarray(indices, dtype=np.int64)
    blocks: list[list[tuple[int, int, int]]] = []
    for label in range(len(FACTORS)):
        examples = allocation[label]
        n_blocks = examples // 2
        positive_pool = indices[targets[indices, label] > 0]
        negative_pool = indices[targets[indices, label] <= 0]
        if not len(positive_pool) or not len(negative_pool):
            raise RuntimeError(
                f"Factor {FACTORS[label]!r} lacks a positive or negative training pair"
            )

        # Repeated annotations remain a weak salience signal for sampling, but
        # the final target remains binary multi-label presence.
        salience = 1.0 + 0.30 * np.log1p(
            np.maximum(counts[positive_pool, label] - 1.0, 0.0)
        )
        salience = salience / salience.sum()
        positives = rng.choice(positive_pool, size=n_blocks, replace=True, p=salience)

        confusion_ids = [FACTOR_TO_ID[name.casefold()]
                         for name in SPECS[label]["confusions"]]
        if confusion_ids:
            confusion_mask = targets[negative_pool][:, confusion_ids].sum(1) > 0
            hard_pool = negative_pool[confusion_mask]
        else:
            hard_pool = np.empty(0, dtype=np.int64)

        for positive in positives:
            use_hard = len(hard_pool) and rng.random() < hard_negative_share
            pool = hard_pool if use_hard else negative_pool
            negative = int(rng.choice(pool))
            block = [(int(positive), label, 1), (negative, label, 0)]
            if rng.random() < 0.5:
                block.reverse()
            blocks.append(block)

    rng.shuffle(blocks)
    return [item for block in blocks for item in block]


def make_inference_pairs(indices: np.ndarray) -> list[tuple[int, int, int]]:
    return [
        (int(index), label, -1)
        for index in np.asarray(indices, dtype=np.int64)
        for label in range(len(FACTORS))
    ]


class PairEncoder:
    def __init__(self, tokenizer, posts, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.suffix = tokenizer.encode(DECISION_SUFFIX, add_special_tokens=False)
        self.prefixes = []
        for label in range(len(FACTORS)):
            body = (
                SYSTEM_PREFIX
                + f"<Instruct>: {INSTRUCTION}\n"
                + f"<Query>:\n{factor_query(label)}\n"
                + "<Document>:\n"
            )
            self.prefixes.append(tokenizer.encode(body, add_special_tokens=False))
        self.min_post_budget = min(
            self.max_length - len(prefix) - len(self.suffix)
            for prefix in self.prefixes
        )
        if self.min_post_budget < 384:
            raise ValueError(
                f"Factor definitions leave only {self.min_post_budget} post tokens"
            )
        self.post_tokens = []
        self.cue_tokens = []
        for text in tqdm(posts, desc="Tokenizing Reddit posts once"):
            text = str(text)
            self.post_tokens.append(tokenizer.encode(text, add_special_tokens=False))
            snippets = [
                text[max(0, match.start() - 150):match.end() + 220]
                for match in list(GENERAL_CUES.finditer(text))[:12]
            ]
            cue_text = "\n[...relevant context...]\n".join(snippets)
            self.cue_tokens.append(
                tokenizer.encode(cue_text, add_special_tokens=False)
                if cue_text else []
            )
        print(json.dumps({
            "max_length": self.max_length,
            "longest_factor_prefix": max(map(len, self.prefixes)),
            "decision_suffix_tokens": len(self.suffix),
            "minimum_post_budget": self.min_post_budget,
        }), flush=True)

    def _fit_post(self, post_index: int, budget: int) -> list[int]:
        tokens = self.post_tokens[post_index]
        if len(tokens) <= budget:
            return tokens
        cues = self.cue_tokens[post_index]
        cue_budget = min(len(cues), int(0.30 * budget))
        remaining = budget - cue_budget
        head = int(0.55 * remaining)
        tail = remaining - head
        result = tokens[:head]
        if cue_budget:
            result += cues[:cue_budget]
        if tail:
            result += tokens[-tail:]
        return result[:budget]

    def encode(self, post_index: int, label: int) -> torch.Tensor:
        prefix = self.prefixes[label]
        budget = self.max_length - len(prefix) - len(self.suffix)
        post = self._fit_post(post_index, budget)
        return torch.tensor(prefix + post + self.suffix, dtype=torch.long)


class FactorPairDataset(Dataset):
    def __init__(self, encoder: PairEncoder, pairs):
        self.encoder = encoder
        self.pairs = list(pairs)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        post_index, label, target = self.pairs[index]
        return {
            "input_ids": self.encoder.encode(post_index, label),
            "post_index": post_index,
            "factor_index": label,
            "target": target,
        }


def collate_pairs(rows, pad_token_id: int):
    longest = max(len(row["input_ids"]) for row in rows)
    input_ids = torch.full(
        (len(rows), longest), int(pad_token_id), dtype=torch.long
    )
    attention_mask = torch.zeros((len(rows), longest), dtype=torch.long)
    for row_index, row in enumerate(rows):
        length = len(row["input_ids"])
        # Left padding preserves the official reranker convention and ensures
        # the final position is always the constrained decision position.
        input_ids[row_index, -length:] = row["input_ids"]
        attention_mask[row_index, -length:] = 1
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "post_index": torch.tensor([row["post_index"] for row in rows]),
        "factor_index": torch.tensor([row["factor_index"] for row in rows]),
        "target": torch.tensor([row["target"] for row in rows], dtype=torch.long),
    }


def make_loader(dataset, pad_token_id, batch_size=2):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=lambda rows: collate_pairs(rows, pad_token_id),
    )


def _single_token_id(tokenizer, text: str) -> int:
    direct = tokenizer.convert_tokens_to_ids(text)
    if direct is not None and direct != tokenizer.unk_token_id:
        return int(direct)
    encoded = tokenizer.encode(text, add_special_tokens=False)
    if len(encoded) != 1:
        raise ValueError(f"Reranker decision {text!r} is not one token: {encoded}")
    return int(encoded[0])


def load_model(args):
    model_name = MODEL_CHOICES[args.model]
    token = os.environ.get("HF_TOKEN") or None
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    compute_dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )
    max_memory = {index: "14500MiB" for index in range(torch.cuda.device_count())}
    max_memory["cpu"] = "26GiB"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=token,
        quantization_config=quantization,
        device_map="auto",
        max_memory=max_memory,
        dtype=compute_dtype,
        low_cpu_mem_usage=True,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    try:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
    except TypeError:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True
        )
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.rank,
        lora_alpha=2 * args.rank,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    ))
    model.print_trainable_parameters()
    input_device = model.get_input_embeddings().weight.device
    output_device = model.get_output_embeddings().weight.device
    decision_ids = torch.tensor(
        [_single_token_id(tokenizer, "no"), _single_token_id(tokenizer, "yes")],
        device=output_device,
        dtype=torch.long,
    )
    return model, tokenizer, input_device, output_device, decision_ids


def decision_logits(
    model,
    input_ids,
    attention_mask,
    input_device,
    output_device,
    decision_ids,
):
    """Compute only the final yes/no logits instead of a full token LM loss."""
    base = model.get_base_model()
    hidden = base.model(
        input_ids=input_ids.to(input_device, non_blocking=True),
        attention_mask=attention_mask.to(input_device, non_blocking=True),
        use_cache=False,
    ).last_hidden_state[:, -1]
    vocabulary_logits = base.lm_head(hidden.to(output_device))
    return vocabulary_logits.index_select(1, decision_ids)


def train_model(
    model,
    tokenizer,
    encoder,
    train_indices,
    targets,
    counts,
    input_device,
    output_device,
    decision_ids,
    args,
):
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = AdamW(
        parameters, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    batches_per_epoch = math.ceil(args.pairs_per_epoch / args.batch_size)
    updates = math.ceil(batches_per_epoch / args.accumulation) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        max(1, int(args.warmup_ratio * updates)),
        max(1, updates),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    history = []
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(1, args.epochs + 1):
        pairs = make_training_pairs(
            train_indices,
            targets,
            counts,
            args.pairs_per_epoch,
            args.protective_share,
            args.hard_negative_share,
            seed=SEED + 1000 * args.fold + epoch,
        )
        dataset = FactorPairDataset(encoder, pairs)
        loader = make_loader(dataset, tokenizer.pad_token_id, args.batch_size)
        model.train()
        losses = []
        classification_losses = []
        ranking_losses = []
        progress = tqdm(loader, desc=f"Pairwise reranker epoch {epoch}/{args.epochs}")
        for step, batch in enumerate(progress, 1):
            target = batch["target"].to(output_device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = decision_logits(
                    model,
                    batch["input_ids"],
                    batch["attention_mask"],
                    input_device,
                    output_device,
                    decision_ids,
                )
                classification = F.cross_entropy(logits.float(), target)
                scores = logits[:, 1].float() - logits[:, 0].float()
                same_label = batch["factor_index"][0] == batch["factor_index"][-1]
                has_both = target.min() == 0 and target.max() == 1
                if len(target) == 2 and bool(same_label) and bool(has_both):
                    positive = scores[target == 1].mean()
                    negative = scores[target == 0].mean()
                    ranking = F.softplus(
                        negative - positive + args.ranking_margin
                    )
                else:
                    ranking = scores.sum() * 0.0
                loss = (
                    classification + args.ranking_weight * ranking
                ) / args.accumulation
            scaler.scale(loss).backward()
            losses.append(float(loss.detach()) * args.accumulation)
            classification_losses.append(float(classification.detach()))
            ranking_losses.append(float(ranking.detach()))
            if step % args.accumulation == 0 or step == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= previous_scale:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            progress.set_postfix(loss=f"{np.mean(losses[-40:]):.4f}")
        row = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "classification_loss": float(np.mean(classification_losses)),
            "ranking_loss": float(np.mean(ranking_losses)),
            "training_pairs": len(pairs),
            "risk_pairs": int(sum(label < RISK_COUNT for _, label, _ in pairs)),
            "protective_pairs": int(sum(label >= RISK_COUNT for _, label, _ in pairs)),
        }
        history.append(row)
        print(json.dumps(row), flush=True)
    return history


@torch.no_grad()
def infer_posts(
    model,
    tokenizer,
    encoder,
    indices,
    input_device,
    output_device,
    decision_ids,
    batch_size,
    description,
):
    indices = np.asarray(indices, dtype=np.int64)
    position = {int(global_index): local for local, global_index in enumerate(indices)}
    result = np.zeros((len(indices), len(FACTORS)), dtype=np.float32)
    pairs = make_inference_pairs(indices)
    dataset = FactorPairDataset(encoder, pairs)
    loader = make_loader(dataset, tokenizer.pad_token_id, batch_size)
    model.eval()
    for batch in tqdm(loader, desc=description):
        with torch.autocast("cuda", dtype=torch.float16):
            logits = decision_logits(
                model,
                batch["input_ids"],
                batch["attention_mask"],
                input_device,
                output_device,
                decision_ids,
            )
        probability = torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy()
        for global_index, label, value in zip(
            batch["post_index"].tolist(),
            batch["factor_index"].tolist(),
            probability.tolist(),
        ):
            result[position[int(global_index)], int(label)] = float(value)
    return result


def _rank(values):
    values = np.asarray(values, dtype=np.float32)
    ranked = np.empty_like(values)
    denominator = max(1, len(values) - 1)
    for label in range(values.shape[1]):
        order = np.argsort(values[:, label], kind="stable")
        label_rank = np.empty(len(values), dtype=np.float32)
        label_rank[order] = np.arange(len(values), dtype=np.float32) / denominator
        ranked[:, label] = label_rank
    return ranked


def _topk(score, prevalence, ratio):
    count = max(1, min(len(score), int(round(len(score) * prevalence * ratio))))
    selected = np.argpartition(score, len(score) - count)[len(score) - count:]
    prediction = np.zeros(len(score), dtype=bool)
    prediction[selected] = True
    return prediction


def _decode(probability, targets, membership, ratio):
    prediction = np.zeros_like(targets, dtype=bool)
    for fold in range(N_FOLDS):
        fit = np.flatnonzero(membership != fold)
        valid = np.flatnonzero(membership == fold)
        ranked = _rank(probability[valid])
        prevalence = targets[fit].mean(0)
        for label in range(len(FACTORS)):
            prediction[valid, label] = _topk(
                ranked[:, label], prevalence[label], ratio
            )
    return prediction


def _safe_metric(metric, truth, probability):
    if np.unique(truth).size < 2:
        return None
    return float(metric(truth, probability))


def diagnostics(targets, probability, prediction):
    rows = []
    for label, name in enumerate(FACTORS):
        rows.append({
            "label": name,
            "group": factor_group(label),
            "support": int(targets[:, label].sum()),
            "f1": float(f1_score(
                targets[:, label], prediction[:, label], zero_division=0
            )),
            "roc_auc": _safe_metric(
                roc_auc_score, targets[:, label], probability[:, label]
            ),
            "pr_auc": _safe_metric(
                average_precision_score, targets[:, label], probability[:, label]
            ),
        })
    return rows


def grouped_scores(targets, prediction):
    return {
        "macro_f1": float(f1_score(
            targets, prediction, average="macro", zero_division=0
        )),
        "risk_macro_f1": float(f1_score(
            targets[:, :RISK_COUNT], prediction[:, :RISK_COUNT],
            average="macro", zero_division=0,
        )),
        "protective_macro_f1": float(f1_score(
            targets[:, RISK_COUNT:], prediction[:, RISK_COUNT:],
            average="macro", zero_division=0,
        )),
    }


def model_key(args):
    return f"{args.model}-factor-v3"


def run_fold(args):
    frame, targets, counts = load_train()
    split = np.load(locate("factor_baseline_oof.npz"), allow_pickle=True)
    membership = split["fold_membership"].astype(int)
    fit = np.flatnonzero(membership != args.fold)
    valid = np.flatnonzero(membership == args.fold)
    train_users = set(frame.anon_user_id.iloc[fit])
    valid_users = set(frame.anon_user_id.iloc[valid])
    if train_users & valid_users:
        raise RuntimeError("User leakage detected in packaged factor split")
    print(
        f"MODEL={MODEL_CHOICES[args.model]} fold={args.fold} "
        f"train={len(fit)} valid={len(valid)}; "
        f"decisions at inference={len(valid) * len(FACTORS)}",
        flush=True,
    )
    model, tokenizer, input_device, output_device, decision_ids = load_model(args)
    encoder = PairEncoder(tokenizer, frame.post.tolist(), args.max_length)
    torch.cuda.reset_peak_memory_stats()
    history = train_model(
        model, tokenizer, encoder, fit, targets, counts,
        input_device, output_device, decision_ids, args,
    )
    probability = infer_posts(
        model, tokenizer, encoder, valid,
        input_device, output_device, decision_ids,
        args.inference_batch_size,
        f"Reranker fold {args.fold} inference (24 decisions/post)",
    )
    ranked = _rank(probability)
    prevalence = targets[fit].mean(0)
    prediction = np.zeros_like(targets[valid], dtype=bool)
    for label in range(len(FACTORS)):
        prediction[:, label] = _topk(ranked[:, label], prevalence[label], 1.0)
    baseline_probability = split["probabilities"][valid].astype(np.float32)
    baseline_ranked = _rank(baseline_probability)
    baseline_prediction = np.zeros_like(targets[valid], dtype=bool)
    for label in range(len(FACTORS)):
        baseline_prediction[:, label] = _topk(
            baseline_ranked[:, label], prevalence[label], 1.10
        )
    per_label = diagnostics(targets[valid], probability, prediction)
    baseline_per_label = diagnostics(
        targets[valid], baseline_probability, baseline_prediction
    )
    for reranker_row, baseline_row in zip(per_label, baseline_per_label):
        reranker_row["baseline_f1"] = baseline_row["f1"]
        reranker_row["baseline_roc_auc"] = baseline_row["roc_auc"]
        reranker_row["baseline_pr_auc"] = baseline_row["pr_auc"]
    payload = {
        "stage": "fold",
        "training_version": TRAINING_VERSION,
        "model_key": args.model,
        "model": MODEL_CHOICES[args.model],
        "fold": args.fold,
        "train_posts": len(fit),
        "valid_posts": len(valid),
        "train_users": len(train_users),
        "valid_users": len(valid_users),
        "user_disjoint": True,
        "epochs": args.epochs,
        "max_length": args.max_length,
        "pairs_per_epoch": args.pairs_per_epoch,
        "protective_share": args.protective_share,
        "history": history,
        "packaged_baseline_rank_scores": grouped_scores(
            targets[valid], baseline_prediction
        ),
        "standalone_rank_scores": grouped_scores(targets[valid], prediction),
        "per_label": per_label,
        "peak_allocated_gib": float(
            torch.cuda.max_memory_allocated() / 1024 ** 3
        ),
    }
    stem = f"{model_key(args)}_fold{args.fold}"
    np.savez_compressed(
        WORK / f"{stem}_probabilities.npz",
        global_indices=valid.astype(np.int32),
        probabilities=probability.astype(np.float16),
        truth=targets[valid].astype(np.int8),
        factors=np.asarray(FACTORS, dtype="U"),
    )
    (WORK / f"{stem}_results.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    if args.save_adapter:
        adapter = WORK / f"{stem}_adapter"
        model.save_pretrained(adapter)
        tokenizer.save_pretrained(adapter)
        (adapter / "reranker_manifest.json").write_text(
            json.dumps({
                "training_version": TRAINING_VERSION,
                "model": MODEL_CHOICES[args.model],
                "factors": FACTORS,
                "risk_count": RISK_COUNT,
                "max_length": args.max_length,
            }, indent=2),
            encoding="utf-8",
        )
    print(json.dumps(payload, indent=2), flush=True)


def _select_on_fit(base, reranker, targets, fit, label):
    base_rank = _rank(base[fit])[:, label]
    reranker_rank = _rank(reranker[fit])[:, label]
    prevalence = float(targets[fit, label].mean())
    ratios = (
        (0.60, 0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.30, 1.40)
        if label < RISK_COUNT
        else (0.50, 0.65, 0.80, 0.95, 1.10, 1.25, 1.40, 1.60)
    )
    candidates = []
    for weight in (0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.70, 1.0):
        for ratio in ratios:
            score = (1.0 - weight) * base_rank + weight * reranker_rank
            prediction = _topk(score, prevalence, ratio)
            value = float(f1_score(
                targets[fit, label], prediction, zero_division=0
            ))
            # Conservative penalty: the reranker is admitted only when it
            # supplies enough improvement to justify a new production expert.
            objective = value - 0.004 * weight - 0.0015 * abs(ratio - 1.0)
            candidates.append((objective, value, -weight, weight, ratio))
    _, value, _, weight, ratio = max(candidates)
    return {"weight": float(weight), "ratio": float(ratio), "fit_f1": value}


def summarize_oof(args):
    frame, targets, _ = load_train()
    baseline_file = np.load(locate("factor_baseline_oof.npz"), allow_pickle=True)
    membership = baseline_file["fold_membership"].astype(int)
    baseline_probability = baseline_file["probabilities"].astype(np.float32)
    if not np.array_equal(
        baseline_file["targets"].astype(np.int8), targets.astype(np.int8)
    ):
        raise RuntimeError("Packaged factor targets do not match train.xlsx")

    reranker_probability = np.zeros_like(baseline_probability)
    covered = np.zeros(len(frame), dtype=bool)
    for fold in range(N_FOLDS):
        name = f"{model_key(args)}_fold{fold}_probabilities.npz"
        path = WORK / name
        if not path.exists():
            path = locate(name)
        part = np.load(path, allow_pickle=True)
        indices = part["global_indices"].astype(int)
        if not np.array_equal(
            part["truth"].astype(np.int8), targets[indices].astype(np.int8)
        ):
            raise RuntimeError(f"Fold {fold} truth differs from train.xlsx")
        reranker_probability[indices] = part["probabilities"].astype(np.float32)
        covered[indices] = True
    if not covered.all():
        raise RuntimeError(f"OOF probabilities miss {(~covered).sum()} posts")

    baseline_prediction = _decode(
        baseline_probability, targets, membership, 1.10
    )
    standalone_prediction = _decode(
        reranker_probability, targets, membership, 1.00
    )
    candidate_prediction = np.zeros_like(targets, dtype=bool)
    fold_selections = []
    for fold in range(N_FOLDS):
        fit = np.flatnonzero(membership != fold)
        valid = np.flatnonzero(membership == fold)
        base_valid = _rank(baseline_probability[valid])
        reranker_valid = _rank(reranker_probability[valid])
        selections = []
        for label in range(len(FACTORS)):
            selected = _select_on_fit(
                baseline_probability, reranker_probability, targets, fit, label
            )
            score = (
                (1.0 - selected["weight"]) * base_valid[:, label]
                + selected["weight"] * reranker_valid[:, label]
            )
            candidate_prediction[valid, label] = _topk(
                score, targets[fit, label].mean(), selected["ratio"]
            )
            selections.append({
                "label": FACTORS[label],
                "group": factor_group(label),
                **selected,
            })
        fold_selections.append({
            "fold": fold,
            "baseline": grouped_scores(
                targets[valid], baseline_prediction[valid]
            ),
            "reranker": grouped_scores(
                targets[valid], standalone_prediction[valid]
            ),
            "candidate": grouped_scores(
                targets[valid], candidate_prediction[valid]
            ),
            "selections": selections,
        })

    per_label = []
    production_parameters = []
    for label, name in enumerate(FACTORS):
        baseline_f1 = float(f1_score(
            targets[:, label], baseline_prediction[:, label], zero_division=0
        ))
        candidate_f1 = float(f1_score(
            targets[:, label], candidate_prediction[:, label], zero_division=0
        ))
        choices = [row["selections"][label] for row in fold_selections]
        nonzero = [choice for choice in choices if choice["weight"] > 0]
        accepted = bool(candidate_f1 > baseline_f1 and len(nonzero) >= 3)
        if accepted:
            weight = float(np.median([choice["weight"] for choice in nonzero]))
            ratio = float(np.median([choice["ratio"] for choice in nonzero]))
        else:
            weight = 0.0
            ratio = float(np.median([
                choice["ratio"] for choice in choices if choice["weight"] == 0
            ] or [1.10]))
        row = {
            "label": name,
            "group": factor_group(label),
            "support": int(targets[:, label].sum()),
            "baseline_f1": baseline_f1,
            "reranker_f1": float(f1_score(
                targets[:, label], standalone_prediction[:, label],
                zero_division=0,
            )),
            "candidate_f1": candidate_f1,
            "delta": candidate_f1 - baseline_f1,
            "reranker_roc_auc": _safe_metric(
                roc_auc_score, targets[:, label], reranker_probability[:, label]
            ),
            "reranker_pr_auc": _safe_metric(
                average_precision_score,
                targets[:, label],
                reranker_probability[:, label],
            ),
            "selected_folds": len(nonzero),
            "accepted": accepted,
        }
        per_label.append(row)
        production_parameters.append({
            "label": name,
            "group": factor_group(label),
            "weight": weight,
            "ratio": ratio,
            "accepted": accepted,
        })

    payload = {
        "stage": "summarize-oof",
        "training_version": TRAINING_VERSION,
        "model": MODEL_CHOICES[args.model],
        "posts": len(frame),
        "folds": N_FOLDS,
        "baseline": grouped_scores(targets, baseline_prediction),
        "reranker_standalone": grouped_scores(targets, standalone_prediction),
        "nested_candidate": grouped_scores(targets, candidate_prediction),
        "fold_results": [
            {key: value for key, value in row.items() if key != "selections"}
            for row in fold_selections
        ],
        "per_label": per_label,
        "production_parameters": production_parameters,
    }
    stem = model_key(args)
    (WORK / f"{stem}_oof_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        WORK / f"{stem}_oof_probabilities.npz",
        probabilities=reranker_probability.astype(np.float16),
        targets=targets.astype(np.int8),
        fold_membership=membership.astype(np.int8),
        row_id=frame.row_id.astype(str).to_numpy(dtype="U"),
        factors=np.asarray(FACTORS, dtype="U"),
    )
    print(json.dumps(payload, indent=2), flush=True)


def train_full(args):
    train_frame, targets, counts = load_train()
    test_frame = load_test()
    combined_posts = train_frame.post.tolist() + test_frame.post.tolist()
    model, tokenizer, input_device, output_device, decision_ids = load_model(args)
    encoder = PairEncoder(tokenizer, combined_posts, args.max_length)
    history = train_model(
        model,
        tokenizer,
        encoder,
        np.arange(len(train_frame)),
        targets,
        counts,
        input_device,
        output_device,
        decision_ids,
        args,
    )
    test_indices = np.arange(len(train_frame), len(combined_posts))
    probability = infer_posts(
        model,
        tokenizer,
        encoder,
        test_indices,
        input_device,
        output_device,
        decision_ids,
        args.inference_batch_size,
        "Leaderboard inference (24 decisions/post)",
    )
    stem = model_key(args)
    np.savez_compressed(
        WORK / f"{stem}_test_probabilities.npz",
        row_id=test_frame.row_id.astype(str).to_numpy(dtype="U"),
        probabilities=probability.astype(np.float16),
        factors=np.asarray(FACTORS, dtype="U"),
        factor_group=np.asarray(
            [factor_group(label) for label in range(len(FACTORS))], dtype="U"
        ),
    )
    payload = {
        "stage": "full",
        "training_version": TRAINING_VERSION,
        "model": MODEL_CHOICES[args.model],
        "train_posts": len(train_frame),
        "test_posts": len(test_frame),
        "factor_decisions": len(test_frame) * len(FACTORS),
        "epochs": args.epochs,
        "max_length": args.max_length,
        "pairs_per_epoch": args.pairs_per_epoch,
        "protective_share": args.protective_share,
        "history": history,
        "mean_probabilities": {
            name: float(probability[:, label].mean())
            for label, name in enumerate(FACTORS)
        },
    }
    (WORK / f"{stem}_full_results.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    if args.save_adapter:
        adapter = WORK / f"{stem}_full_adapter"
        model.save_pretrained(adapter)
        tokenizer.save_pretrained(adapter)
        (adapter / "reranker_manifest.json").write_text(
            json.dumps({
                "training_version": TRAINING_VERSION,
                "model": MODEL_CHOICES[args.model],
                "factors": FACTORS,
                "risk_count": RISK_COUNT,
                "max_length": args.max_length,
                "instruction": INSTRUCTION,
            }, indent=2),
            encoding="utf-8",
        )
        print(f"Saved reusable adapter: {adapter}", flush=True)
    print(json.dumps(payload, indent=2), flush=True)


def preflight(args):
    frame, targets, counts = load_train()
    split = np.load(locate("factor_baseline_oof.npz"), allow_pickle=True)
    membership = split["fold_membership"].astype(int)
    if not np.array_equal(
        split["targets"].astype(np.int8), targets.astype(np.int8)
    ):
        raise RuntimeError("Packaged targets differ from train.xlsx")
    groups = frame.anon_user_id.astype(str).to_numpy()
    leakage = []
    for fold in range(N_FOLDS):
        fit = np.flatnonzero(membership != fold)
        valid = np.flatnonzero(membership == fold)
        leakage.append(len(set(groups[fit]) & set(groups[valid])))

    token = os.environ.get("HF_TOKEN") or None
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_CHOICES[args.model], token=token
    )
    suffix_length = len(tokenizer.encode(
        DECISION_SUFFIX, add_special_tokens=False
    ))
    prefix_lengths = []
    for label in range(len(FACTORS)):
        body = (
            SYSTEM_PREFIX
            + f"<Instruct>: {INSTRUCTION}\n"
            + f"<Query>:\n{factor_query(label)}\n"
            + "<Document>:\n"
        )
        prefix_lengths.append(len(tokenizer.encode(
            body, add_special_tokens=False
        )))
    payload = {
        "training_version": TRAINING_VERSION,
        "model": MODEL_CHOICES[args.model],
        "preflight_passed": bool(
            not any(leakage)
            and targets.shape == (len(frame), len(FACTORS))
            and args.max_length - max(prefix_lengths) - suffix_length >= 384
        ),
        "posts": len(frame),
        "users": int(len(set(groups))),
        "targets_shape": list(targets.shape),
        "risk_labels": len(RISK_INDICES),
        "protective_labels": len(PROTECTIVE_INDICES),
        "risk_positive_annotations": int(targets[:, :RISK_COUNT].sum()),
        "protective_positive_annotations": int(targets[:, RISK_COUNT:].sum()),
        "duplicate_factor_occurrences": int((counts > 1).sum()),
        "fold_sizes": [int((membership == fold).sum()) for fold in range(N_FOLDS)],
        "user_overlap_by_fold": leakage,
        "factor_support": {
            name: {
                "group": factor_group(label),
                "support": int(targets[:, label].sum()),
            }
            for label, name in enumerate(FACTORS)
        },
        "max_factor_query_prefix_tokens": max(prefix_lengths),
        "decision_suffix_tokens": suffix_length,
        "minimum_post_budget": (
            args.max_length - max(prefix_lengths) - suffix_length
        ),
        "planned_pairs_per_epoch": args.pairs_per_epoch,
        "planned_protective_share": args.protective_share,
        "fold_inference_decisions": [
            int((membership == fold).sum() * len(FACTORS))
            for fold in range(N_FOLDS)
        ],
    }
    print(json.dumps(payload, indent=2), flush=True)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=["preflight", "fold", "summarize-oof", "full"],
        default="fold",
    )
    parser.add_argument(
        "--model", choices=sorted(MODEL_CHOICES),
        default="qwen3-reranker-8b",
    )
    parser.add_argument("--fold", type=int, choices=range(N_FOLDS), default=0)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=PRESET["max_length"])
    parser.add_argument("--rank", type=int, default=PRESET["rank"])
    parser.add_argument("--batch-size", type=int, default=PRESET["batch_size"])
    parser.add_argument(
        "--inference-batch-size", type=int, default=PRESET["batch_size"]
    )
    parser.add_argument(
        "--accumulation", type=int, default=PRESET["accumulation"]
    )
    parser.add_argument(
        "--learning-rate", type=float, default=PRESET["learning_rate"]
    )
    parser.add_argument(
        "--weight-decay", type=float, default=PRESET["weight_decay"]
    )
    parser.add_argument(
        "--warmup-ratio", type=float, default=PRESET["warmup_ratio"]
    )
    parser.add_argument(
        "--pairs-per-epoch", type=int, default=PRESET["pairs_per_epoch"]
    )
    parser.add_argument(
        "--protective-share", type=float, default=PRESET["protective_share"]
    )
    parser.add_argument(
        "--hard-negative-share", type=float,
        default=PRESET["hard_negative_share"],
    )
    parser.add_argument(
        "--ranking-weight", type=float, default=PRESET["ranking_weight"]
    )
    parser.add_argument(
        "--ranking-margin", type=float, default=PRESET["ranking_margin"]
    )
    parser.add_argument("--save-adapter", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    if not 0.05 <= args.protective_share <= 0.60:
        raise ValueError("--protective-share must be between 0.05 and 0.60")
    if args.pairs_per_epoch < 4 * len(FACTORS):
        raise ValueError("--pairs-per-epoch is too small for balanced pairs")
    if (
        not torch.cuda.is_available()
        and args.stage not in ("preflight", "summarize-oof")
    ):
        raise RuntimeError("Enable a Kaggle GPU accelerator before training")
    seed_everything(SEED + args.fold)
    print(json.dumps({
        "torch": torch.__version__,
        "cuda": torch.cuda.is_available(),
        "gpus": [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ],
        "arguments": vars(args),
    }, indent=2), flush=True)
    if args.stage == "preflight":
        preflight(args)
    elif args.stage == "fold":
        run_fold(args)
    elif args.stage == "summarize-oof":
        summarize_oof(args)
    else:
        train_full(args)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
