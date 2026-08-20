"""Kaggle Qwen3-8B QLoRA V2 for 24-label suicide factors.

V2 mirrors the strongest local MentalRoBERTa design: a contextual-definition
global detector is the main head and label-specific token attention is a
learnable residual.  It also uses occurrence-aware asymmetric loss and a
cross-post, per-label ranking objective.  The script produces strict
user-disjoint OOF probabilities before any full-data leaderboard inference.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
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
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from torch.nn.utils.rnn import pad_sequence
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)


ROOT = Path(__file__).resolve().parent
WORK = Path(os.environ.get("KAGGLE_WORKING_DIR", "/kaggle/working"))
if not WORK.exists():
    WORK = ROOT / "working"
WORK.mkdir(parents=True, exist_ok=True)

SEED = 42
N_FOLDS = 5
TRAINING_VERSION = "qwen3-8b-factor-hybrid-v2"
MODEL_KEY = "qwen3-8b-factor-v2"
MODEL_NAME = "Qwen/Qwen3-8B"

FACTORS = [
    "mental health issues",
    "physical health/characteristic",
    "substance use",
    "hopelessness",
    "emotion dysregulation",
    "low self-esteem",
    "poor school performance",
    "low socio-economic status",
    "interpersonal violence",
    "prior self-harm or suicidal thought/attempt",
    "poor social support",
    "interpersonal difficulty",
    "dysfunctional family",
    "exposure to others' suicide",
    "stressful life event",
    "traumatic experience",
    "cognitive deficits",
    "suicide means (with access)",
    "sexual orientation related issues",
    "social support",
    "coping strategy",
    "psychological capital",
    "sense of responsibility",
    "meaning in life",
]
FACTOR_TO_ID = {name.casefold(): i for i, name in enumerate(FACTORS)}

# Compact definitions preserve enough post-token budget while explicitly
# separating the most frequently confused concepts.
DEFINITIONS = [
    "diagnosed or symptomatic depression, anxiety, psychosis, PTSD, eating disorder, or other mental illness",
    "physical illness, disability, pain, body condition, sleep, weight, age, or other bodily characteristic",
    "alcohol, illicit drugs, medication misuse, intoxication, addiction, or withdrawal",
    "the future feels impossible, pointless, unchangeable, or without any way out; not merely temporary sadness",
    "overwhelming or poorly controlled anger, panic, agitation, mood swings, impulsivity, or emotional instability",
    "self-hatred, worthlessness, shame, feeling like a failure or burden, or globally negative self-evaluation",
    "failing grades, inability to study, academic dismissal, or serious school-performance problems",
    "poverty, unemployment, debt, housing or food insecurity, or inability to afford necessities",
    "physical, sexual, emotional, or verbal abuse; assault, bullying, threats, coercion, or partner violence",
    "the author's previous/current self-harm, suicidal thoughts, suicidal behaviour, or suicide attempt",
    "isolation, abandonment, nobody caring/listening, unavailable help, or perceived lack of support",
    "conflict, breakup, rejection, loneliness, difficulty communicating, or problems with peers/partners",
    "family conflict, neglect, abuse, separation, addiction, instability, or harmful family functioning",
    "another person's suicide or attempt affects the author; exclude the author's own attempt",
    "major recent pressure or change such as loss, breakup, bereavement, legal/work/school crisis, or relocation",
    "a disturbing event with lasting psychological impact, abuse history, disaster, violence, or severe loss",
    "confusion, impaired concentration, indecision, distorted thinking, memory difficulty, or cognitive rigidity",
    "the author possesses or can readily access a suicide method such as pills, firearm, rope, height, or vehicle",
    "distress, discrimination, rejection, conflict, or identity struggle specifically related to sexual orientation",
    "actual received/available care, connection, listening, help, companionship, or belonging from other people",
    "an action used to manage distress: help-seeking, therapy, distraction, exercise, hobbies, relaxation, or problem solving",
    "hope, resilience, optimism, confidence, self-efficacy, recovery belief, courage, or emotional inner strength",
    "a felt duty to survive or care for family, children, friends, pets, work, health, or other commitments",
    "purpose, values, goals, reasons for living, motivation, or emotionally significant commitments that make life worthwhile",
]

TAXONOMY = "\n".join(
    f"{i + 1}. {name}: {definition}."
    for i, (name, definition) in enumerate(zip(FACTORS, DEFINITIONS))
)
PREFIX = (
    "Read the CURRENT Reddit post and identify every suicide-related risk or "
    "protective factor expressed by the AUTHOR. A factor must be supported by "
    "the post; do not infer diagnoses or circumstances that are absent. Risk and "
    "protective factors may coexist. Distinguish poor social support from actual "
    "social support, hopelessness from low self-esteem, and the author's own "
    "suicidality from exposure to another person's suicide.\n"
    "FACTOR TAXONOMY:\n" + TAXONOMY + "\nCURRENT POST:\n"
)
SUFFIX = "\nINTERNAL FACTOR REPRESENTATION:"

FACTOR_CUE = re.compile(
    r"depress|anxi|panic|ptsd|psych|bipolar|schizo|therap|medicat|pain|sick|"
    r"hospital|sleep|alcohol|drunk|drug|weed|cocaine|addict|withdraw|hopeless|"
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

PRESET = {
    "max_length": 1536,
    "rank": 16,
    # Two posts are required for the cross-post label-ranking objective.
    # Gradient accumulation keeps the effective batch equal to V1 (16).
    "batch_size": 2,
    "accumulation": 8,
    "backbone_lr": 4e-5,
    "head_lr": 1.5e-4,
    "weight_decay": .08,
    "tail_sampling_alpha": .25,
    "max_sample_weight": 2.0,
    "local_residual_prior": .25,
    "ranking_weight": .12,
    "semantic_anchor_weight": .02,
    "occurrence_alpha": .20,
}


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


def _parse_factors(value) -> tuple[np.ndarray, np.ndarray]:
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
        counts[FACTOR_TO_ID[key]] += 1.0
        target[FACTOR_TO_ID[key]] = 1.0
    return target, counts


def load_train() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    frame = pd.read_excel(locate("train.xlsx"))
    frame["post"] = frame["post"].fillna("").astype(str)
    frame["anon_user_id"] = frame["anon_user_id"].fillna(frame["row_id"]).astype(str)
    parsed = [_parse_factors(value) for value in frame["factors"]]
    targets = np.vstack([row[0] for row in parsed])
    counts = np.vstack([row[1] for row in parsed])
    return frame, targets, counts


def _balanced_post_tokens(tokenizer, text: str, budget: int) -> list[int]:
    text = str(text)
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) <= budget:
        return tokens
    blocks = []
    for match in list(FACTOR_CUE.finditer(text))[:12]:
        blocks.append(text[max(0, match.start() - 140):match.end() + 200])
    cue_tokens = tokenizer.encode("\n[...factor context...]\n".join(blocks),
                                  add_special_tokens=False)
    cue_budget = min(len(cue_tokens), int(.45 * budget))
    side = (budget - cue_budget) // 2
    result = tokens[:side]
    if cue_budget:
        result += cue_tokens[:cue_budget]
    result += tokens[-(budget - len(result)):]
    return result[:budget]


class FactorPromptDataset(Dataset):
    def __init__(self, texts, targets, counts, tokenizer, max_length):
        prefix = tokenizer.encode(PREFIX, add_special_tokens=False)
        suffix = tokenizer.encode(SUFFIX, add_special_tokens=False)
        available = max_length - len(prefix) - len(suffix)
        print(json.dumps({"prefix_tokens": len(prefix), "post_token_budget": available,
                          "max_length": max_length}), flush=True)
        if available < 320:
            raise ValueError("Definition prompt leaves too little room for the Reddit post")
        self.pad_token_id = int(tokenizer.pad_token_id)
        self.rows = []
        for text, target, count in tqdm(
            list(zip(texts, targets, counts)), desc="Tokenizing factor posts"
        ):
            post = _balanced_post_tokens(tokenizer, text, available)
            ids = prefix + post + suffix
            post_mask = [0] * len(prefix) + [1] * len(post) + [0] * len(suffix)
            if not any(post_mask):
                post_mask[-1] = 1
            self.rows.append((torch.tensor(ids, dtype=torch.long),
                              torch.tensor(post_mask, dtype=torch.bool),
                              torch.tensor(target, dtype=torch.float32),
                              torch.tensor(count, dtype=torch.float32)))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        ids, post_mask, target, count = self.rows[index]
        return {
            "input_ids": ids, "post_mask": post_mask,
            "targets": target, "factor_counts": count,
        }


def _collate(rows, pad_token_id):
    ids = pad_sequence([row["input_ids"] for row in rows], batch_first=True,
                       padding_value=pad_token_id)
    post_mask = pad_sequence([row["post_mask"] for row in rows], batch_first=True,
                             padding_value=False)
    return {
        "input_ids": ids,
        "attention_mask": ids.ne(pad_token_id).long(),
        "post_mask": post_mask,
        "targets": torch.stack([row["targets"] for row in rows]),
        "factor_counts": torch.stack([row["factor_counts"] for row in rows]),
    }


class HybridFactorHead(nn.Module):
    """Global semantic classifier plus gated label-attention residual.

    V1 forced every decision through label-token attention.  The accepted
    MentalRoBERTa system is stronger because it combines a document-level head
    with a label-local head.  V2 ports that structure to Qwen and initialises
    all label-facing parameters from contextual definition representations.
    """
    def __init__(self, semantic_vectors: torch.Tensor, prevalence: np.ndarray):
        super().__init__()
        vectors = F.normalize(semantic_vectors.float(), dim=-1)
        self.register_buffer("semantic_anchor", vectors.clone())
        self.queries = nn.Parameter(vectors.clone())
        self.local_outputs = nn.Parameter(vectors.clone())
        self.global_outputs = nn.Parameter(vectors.clone())
        prior = np.clip(np.asarray(prevalence, dtype=np.float32), .002, .998)
        # A softened prior stabilises rare labels without forcing the grossly
        # over-confident V1 probability scale.
        self.bias = nn.Parameter(
            .25 * torch.tensor(np.log(prior / (1. - prior)), dtype=torch.float32)
        )
        self.global_logit_scale = nn.Parameter(torch.tensor(math.log(8.0)))
        self.local_logit_scale = nn.Parameter(torch.tensor(math.log(6.0)))
        gate = float(PRESET["local_residual_prior"])
        self.local_gate_logit = nn.Parameter(torch.full(
            (len(FACTORS),), math.log(gate / (1.0 - gate)), dtype=torch.float32
        ))
        self.dropout = nn.Dropout(.15)

    def forward(self, hidden, attention_mask, post_mask, return_parts=False):
        hidden32 = hidden.float()
        normal_hidden = F.normalize(hidden32, dim=-1)
        queries = F.normalize(self.queries, dim=-1)
        score = torch.einsum("btd,ld->btl", normal_hidden, queries) * 8.0
        # The definition prompt conditions every token representation, but the
        # label queries may attend only to CURRENT POST tokens. Otherwise each
        # query would trivially attend to its own definition in the prefix.
        score = score.masked_fill(~post_mask.bool().unsqueeze(-1), -1e4)
        attention = torch.softmax(score, dim=1)
        context = torch.einsum("btl,btd->bld", attention, hidden32)
        context = F.normalize(self.dropout(context), dim=-1)

        # Pool CURRENT POST tokens only.  V1 used the final non-padding token,
        # which is the prompt suffix rather than a post token.
        post_float = post_mask.unsqueeze(-1).to(hidden32.dtype)
        post_mean = (hidden32 * post_float).sum(1) / post_float.sum(1).clamp_min(1.0)
        positions = torch.arange(hidden32.size(1), device=hidden32.device)
        last_post = torch.where(
            post_mask.bool(), positions.unsqueeze(0),
            torch.zeros_like(post_mask, dtype=positions.dtype)
        ).max(1).values
        rows = torch.arange(hidden32.size(0), device=hidden32.device)
        post_last = hidden32[rows, last_post]
        global_context = F.normalize(
            self.dropout(.65 * post_mean + .35 * post_last), dim=-1
        )

        local_outputs = F.normalize(self.local_outputs, dim=-1)
        global_outputs = F.normalize(self.global_outputs, dim=-1)
        local_scale = self.local_logit_scale.exp().clamp(2.0, 20.0)
        global_scale = self.global_logit_scale.exp().clamp(2.0, 20.0)
        local_logits = local_scale * (
            context * local_outputs.unsqueeze(0)
        ).sum(-1)
        global_logits = global_scale * torch.einsum(
            "bd,ld->bl", global_context, global_outputs
        )
        gate = torch.sigmoid(self.local_gate_logit).unsqueeze(0)
        logits = global_logits + gate * local_logits + self.bias
        if return_parts:
            return logits, global_logits + self.bias, local_logits, global_context
        return logits


def _semantic_vectors(model, tokenizer, output_device):
    """Contextually encode definitions instead of averaging input embeddings."""
    base = model.get_base_model()
    input_device = model.get_input_embeddings().weight.device
    vectors = []
    was_training = model.training
    model.eval()
    for name, definition in zip(FACTORS, DEFINITIONS):
        encoded = tokenizer(
            "Factor definition — " + name + ": " + definition,
            add_special_tokens=False, truncation=True, max_length=112,
            return_tensors="pt",
        )
        ids = encoded["input_ids"].to(input_device)
        mask = encoded["attention_mask"].to(input_device)
        with torch.no_grad():
            hidden = base.model(
                input_ids=ids, attention_mask=mask, use_cache=False,
            ).last_hidden_state.float().to(output_device)
        out_mask = mask.to(output_device).unsqueeze(-1).float()
        vector = (hidden * out_mask).sum(1).squeeze(0) / out_mask.sum().clamp_min(1.0)
        vectors.append(vector)
    if was_training:
        model.train()
    return torch.stack(vectors)


def load_model(prevalence):
    token = os.environ.get("HF_TOKEN") or None
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=token)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    quantization = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
    )
    max_memory = {index: "14500MiB" for index in range(torch.cuda.device_count())}
    max_memory["cpu"] = "26GiB"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, token=token, quantization_config=quantization,
        device_map="auto", max_memory=max_memory, dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    try:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
    except TypeError:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=PRESET["rank"],
        lora_alpha=2 * PRESET["rank"], lora_dropout=.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    ))
    model.print_trainable_parameters()
    input_device = model.get_input_embeddings().weight.device
    output_device = model.get_output_embeddings().weight.device
    head = HybridFactorHead(
        _semantic_vectors(model, tokenizer, output_device), prevalence,
    ).to(output_device)
    return model, head, tokenizer, input_device, output_device


def factor_logits(model, head, input_ids, attention_mask, post_mask,
                  input_device, output_device, return_parts=False):
    base = model.get_base_model()
    hidden = base.model(
        input_ids=input_ids.to(input_device, non_blocking=True),
        attention_mask=attention_mask.to(input_device, non_blocking=True),
        use_cache=False,
    ).last_hidden_state
    return head(
        hidden.to(output_device), attention_mask.to(output_device),
        post_mask.to(output_device), return_parts=return_parts,
    )


def _loader(dataset, indices, shuffle=False, sample_weights=None):
    subset = Subset(dataset, list(map(int, indices)))
    sampler = None
    if sample_weights is not None:
        generator = torch.Generator().manual_seed(SEED)
        sampler = WeightedRandomSampler(
            torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights), replacement=True, generator=generator,
        )
        shuffle = False
    return DataLoader(
        subset, batch_size=PRESET["batch_size"], shuffle=shuffle, sampler=sampler,
        num_workers=0, pin_memory=True,
        collate_fn=lambda rows: _collate(rows, dataset.pad_token_id),
    )


def _sample_weights(targets):
    support = targets.sum(0).clip(min=1.0)
    common = float(np.median(support))
    rarity = np.sqrt(common / support).clip(min=1.0)
    strength = (targets * rarity[None, :]).max(1)
    weights = 1.0 + PRESET["tail_sampling_alpha"] * (strength - 1.0)
    return np.clip(weights, 1.0, PRESET["max_sample_weight"])


def grouped_asymmetric_loss(logits, targets, positive_weight, factor_counts=None):
    """Occurrence-aware ASL used by the strongest local factor model."""
    def group(group_logits, group_targets, group_weight, group_counts):
        positive = torch.sigmoid(group_logits.float())
        negative = 1.0 - positive
        negative = (negative + .05).clamp(max=1.0)
        log_loss = group_targets * torch.log(positive.clamp_min(1e-8))
        log_loss += (1.0 - group_targets) * torch.log(negative.clamp_min(1e-8))
        pt = positive * group_targets + negative * (1.0 - group_targets)
        gamma = 4.0 * (1.0 - group_targets)
        weights = 1.0 + (group_weight.unsqueeze(0) - 1.0) * group_targets
        if group_counts is not None:
            repeats = (group_counts - 1.0).clamp_min(0.0)
            occurrence = 1.0 + PRESET["occurrence_alpha"] * torch.log1p(repeats)
            occurrence = occurrence.clamp(max=2.5)
            weights = weights * (1.0 + (occurrence - 1.0) * group_targets)
        return -(log_loss * torch.pow(1.0 - pt, gamma) * weights).mean()
    risk_counts = None if factor_counts is None else factor_counts[:, :19]
    protective_counts = None if factor_counts is None else factor_counts[:, 19:]
    risk = group(
        logits[:, :19], targets[:, :19], positive_weight[:19], risk_counts
    )
    protective = group(
        logits[:, 19:], targets[:, 19:], positive_weight[19:], protective_counts
    )
    return .5 * (risk + protective)


def cross_post_label_ranking_loss(logits, targets, positive_weight):
    """Rank positive posts above negative posts for the same factor."""
    values = []
    for label in range(logits.size(1)):
        positive = logits[:, label][targets[:, label] > 0]
        negative = logits[:, label][targets[:, label] <= 0]
        if len(positive) and len(negative):
            pair = F.softplus(
                negative.unsqueeze(1) - positive.unsqueeze(0) + .10
            ).mean()
            values.append(pair * positive_weight[label].sqrt().clamp(max=2.5))
    return torch.stack(values).mean() if values else logits.sum() * 0.0


def semantic_anchor_loss(head):
    anchor = F.normalize(head.semantic_anchor, dim=-1)
    return torch.stack([
        (1.0 - (F.normalize(value, dim=-1) * anchor).sum(-1)).mean()
        for value in (head.queries, head.local_outputs, head.global_outputs)
    ]).mean()


def train(model, head, dataset, indices, all_targets, input_device,
          output_device, epochs):
    train_targets = all_targets[np.asarray(indices)]
    support = train_targets.sum(0)
    positive_weight = np.sqrt(
        (len(train_targets) - support) / np.maximum(support, 1.0)
    ).clip(1.0, 8.0)
    positive_weight = torch.tensor(positive_weight, dtype=torch.float32,
                                   device=output_device)
    batches = _loader(
        dataset, indices, sample_weights=_sample_weights(train_targets),
    )
    backbone_parameters = [p for p in model.parameters() if p.requires_grad]
    head_parameters = list(head.parameters())
    optimizer = AdamW([
        {"params": backbone_parameters, "lr": PRESET["backbone_lr"]},
        {"params": head_parameters, "lr": PRESET["head_lr"]},
    ], weight_decay=PRESET["weight_decay"])
    updates = math.ceil(len(batches) / PRESET["accumulation"]) * epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, max(1, int(.06 * updates)), max(1, updates),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    parameters = backbone_parameters + head_parameters
    history = []
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(1, epochs + 1):
        model.train(); head.train(); losses = []
        progress = tqdm(batches, desc=f"Factor QLoRA epoch {epoch}/{epochs}")
        for step, batch in enumerate(progress, 1):
            target = batch["targets"].to(output_device, non_blocking=True)
            factor_counts = batch["factor_counts"].to(
                output_device, non_blocking=True
            )
            with torch.autocast("cuda", dtype=torch.float16):
                logits, global_logits, _, _ = factor_logits(
                    model, head, batch["input_ids"], batch["attention_mask"],
                    batch["post_mask"],
                    input_device, output_device, return_parts=True,
                )
                classification = grouped_asymmetric_loss(
                    logits, target, positive_weight, factor_counts,
                )
                global_auxiliary = grouped_asymmetric_loss(
                    global_logits, target, positive_weight, factor_counts,
                )
                ranking = cross_post_label_ranking_loss(
                    logits.float(), target, positive_weight,
                )
                anchor = semantic_anchor_loss(head)
                loss = (
                    classification + .30 * global_auxiliary
                    + PRESET["ranking_weight"] * ranking
                    + PRESET["semantic_anchor_weight"] * anchor
                ) / PRESET["accumulation"]
            scaler.scale(loss).backward()
            losses.append(float(loss.detach()) * PRESET["accumulation"])
            if step % PRESET["accumulation"] == 0 or step == len(batches):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                previous = scaler.get_scale()
                scaler.step(optimizer); scaler.update()
                if scaler.get_scale() >= previous:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            progress.set_postfix(loss=f"{np.mean(losses[-50:]):.4f}")
        row = {"epoch": epoch, "loss": float(np.mean(losses))}
        history.append(row); print(json.dumps(row), flush=True)
    return history


@torch.no_grad()
def infer(model, head, dataset, indices, input_device, output_device, description):
    model.eval(); head.eval(); result = []
    for batch in tqdm(_loader(dataset, indices), desc=description):
        with torch.autocast("cuda", dtype=torch.float16):
            logits = factor_logits(
                model, head, batch["input_ids"], batch["attention_mask"],
                batch["post_mask"],
                input_device, output_device,
            )
        result.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.vstack(result).astype(np.float32)


def _rank(values):
    values = np.asarray(values, dtype=np.float32)
    result = np.empty_like(values)
    denominator = max(1, len(values) - 1)
    for label in range(values.shape[1]):
        order = np.argsort(values[:, label], kind="stable")
        ranks = np.empty(len(values), dtype=np.float32)
        ranks[order] = np.arange(len(values), dtype=np.float32) / denominator
        result[:, label] = ranks
    return result


def _topk(score, prevalence, ratio):
    count = max(1, min(len(score), int(round(len(score) * prevalence * ratio))))
    chosen = np.argpartition(score, len(score) - count)[len(score) - count:]
    prediction = np.zeros(len(score), dtype=bool); prediction[chosen] = True
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
                ranked[:, label], prevalence[label], ratio,
            )
    return prediction


def _safe_metric(function, truth, probability):
    return None if np.unique(truth).size < 2 else float(function(truth, probability))


def _diagnostics(targets, probability, prediction):
    rows = []
    for label, name in enumerate(FACTORS):
        rows.append({
            "label": name,
            "support": int(targets[:, label].sum()),
            "f1": float(f1_score(targets[:, label], prediction[:, label],
                                 zero_division=0)),
            "roc_auc": _safe_metric(roc_auc_score, targets[:, label],
                                    probability[:, label]),
            "pr_auc": _safe_metric(average_precision_score, targets[:, label],
                                   probability[:, label]),
        })
    return rows


def run_fold(args):
    frame, targets, counts = load_train()
    split = np.load(locate("factor_baseline_oof.npz"), allow_pickle=True)
    membership = split["fold_membership"].astype(int)
    fit = np.flatnonzero(membership != args.fold)
    valid = np.flatnonzero(membership == args.fold)
    if set(frame.anon_user_id.iloc[fit]) & set(frame.anon_user_id.iloc[valid]):
        raise RuntimeError("User leakage detected in packaged factor split")
    print(f"MODEL={MODEL_NAME} factor fold={args.fold} train={len(fit)} "
          f"valid={len(valid)}", flush=True)
    model, head, tokenizer, input_device, output_device = load_model(
        targets[fit].mean(0),
    )
    dataset = FactorPromptDataset(
        frame.post.tolist(), targets, counts, tokenizer, PRESET["max_length"],
    )
    torch.cuda.reset_peak_memory_stats()
    history = train(model, head, dataset, fit, targets, input_device,
                    output_device, args.epochs)
    probability = infer(model, head, dataset, valid, input_device, output_device,
                        f"Factor fold {args.fold} inference")
    prediction = np.zeros_like(targets[valid], dtype=bool)
    ranked = _rank(probability); prevalence = targets[fit].mean(0)
    for label in range(len(FACTORS)):
        prediction[:, label] = _topk(ranked[:, label], prevalence[label], 1.0)
    payload = {
        "stage": "fold", "training_version": TRAINING_VERSION,
        "model": MODEL_NAME, "fold": args.fold,
        "train_posts": len(fit), "valid_posts": len(valid),
        "train_users": int(frame.anon_user_id.iloc[fit].nunique()),
        "valid_users": int(frame.anon_user_id.iloc[valid].nunique()),
        "user_disjoint": True, "epochs": args.epochs,
        "max_length": PRESET["max_length"], "history": history,
        "standalone_rank_macro_f1": float(f1_score(
            targets[valid], prediction, average="macro", zero_division=0)),
        "per_label": _diagnostics(targets[valid], probability, prediction),
        "peak_allocated_gib": float(torch.cuda.max_memory_allocated() / 1024 ** 3),
    }
    stem = f"{MODEL_KEY}_fold{args.fold}"
    np.savez_compressed(
        WORK / f"{stem}_probabilities.npz",
        global_indices=valid, probabilities=probability.astype(np.float16),
        truth=targets[valid].astype(np.int8),
    )
    (WORK / f"{stem}_results.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8",
    )
    if args.save_adapter:
        adapter = WORK / f"{stem}_adapter"
        model.save_pretrained(adapter); tokenizer.save_pretrained(adapter)
        torch.save({"state_dict": {k: v.detach().cpu() for k, v in head.state_dict().items()},
                    "factors": FACTORS}, adapter / "factor_head.pt")
    print(json.dumps(payload, indent=2), flush=True)


def _select_on_fit(base, qwen, targets, fit, label):
    base_rank = _rank(base[fit])[:, label]
    qwen_rank = _rank(qwen[fit])[:, label]
    prevalence = float(targets[fit, label].mean())
    rows = []
    for weight in (0.0, .10, .20, .30, .40, .50, .70, 1.0):
        for ratio in (.60, .70, .80, .90, 1.0, 1.10, 1.20, 1.30, 1.40):
            score = (1.0 - weight) * base_rank + weight * qwen_rank
            prediction = _topk(score, prevalence, ratio)
            value = float(f1_score(targets[fit, label], prediction, zero_division=0))
            objective = value - .004 * weight - .002 * abs(ratio - 1.10)
            rows.append((objective, value, -weight, weight, ratio))
    _, value, _, weight, ratio = max(rows)
    return {"weight": float(weight), "ratio": float(ratio), "fit_f1": value}


def summarize_oof(args):
    frame, targets, _ = load_train()
    saved = np.load(locate("factor_baseline_oof.npz"), allow_pickle=True)
    membership = saved["fold_membership"].astype(int)
    base = saved["probabilities"].astype(np.float32)
    if not np.array_equal(saved["targets"].astype(np.int8), targets.astype(np.int8)):
        raise RuntimeError("Packaged factor targets no longer match train.xlsx")
    qwen = np.zeros_like(base); covered = np.zeros(len(frame), dtype=bool)
    for fold in range(N_FOLDS):
        path = WORK / f"{MODEL_KEY}_fold{fold}_probabilities.npz"
        if not path.exists():
            path = locate(path.name)
        part = np.load(path, allow_pickle=True)
        indices = part["global_indices"].astype(int)
        if not np.array_equal(part["truth"].astype(np.int8), targets[indices].astype(np.int8)):
            raise RuntimeError(f"Fold {fold} truth differs from train.xlsx")
        qwen[indices] = part["probabilities"].astype(np.float32)
        covered[indices] = True
    if not covered.all():
        raise RuntimeError(f"OOF probabilities miss {(~covered).sum()} posts")

    baseline = _decode(base, targets, membership, 1.10)
    standalone = _decode(qwen, targets, membership, 1.00)
    candidate = np.zeros_like(targets, dtype=bool)
    selections = []
    for fold in range(N_FOLDS):
        fit = np.flatnonzero(membership != fold)
        valid = np.flatnonzero(membership == fold)
        base_valid = _rank(base[valid]); qwen_valid = _rank(qwen[valid])
        fold_selection = []
        for label in range(len(FACTORS)):
            selected = _select_on_fit(base, qwen, targets, fit, label)
            score = ((1.0 - selected["weight"]) * base_valid[:, label]
                     + selected["weight"] * qwen_valid[:, label])
            candidate[valid, label] = _topk(
                score, targets[fit, label].mean(), selected["ratio"],
            )
            fold_selection.append({"label": FACTORS[label], **selected})
        selections.append({
            "fold": fold,
            "baseline_macro_f1": float(f1_score(
                targets[valid], baseline[valid], average="macro", zero_division=0)),
            "qwen_macro_f1": float(f1_score(
                targets[valid], standalone[valid], average="macro", zero_division=0)),
            "candidate_macro_f1": float(f1_score(
                targets[valid], candidate[valid], average="macro", zero_division=0)),
            "selections": fold_selection,
        })

    per_label = []
    production = []
    for label, name in enumerate(FACTORS):
        old = float(f1_score(targets[:, label], baseline[:, label], zero_division=0))
        new = float(f1_score(targets[:, label], candidate[:, label], zero_division=0))
        choices = [row["selections"][label] for row in selections]
        nonzero = [row for row in choices if row["weight"] > 0]
        accepted = bool(new > old and len(nonzero) >= 3)
        if accepted:
            weight = float(np.median([row["weight"] for row in nonzero]))
            ratio = float(np.median([row["ratio"] for row in nonzero]))
        else:
            weight = 0.0
            ratio = float(np.median([row["ratio"] for row in choices
                                     if row["weight"] == 0] or [1.10]))
        per_label.append({
            "label": name, "support": int(targets[:, label].sum()),
            "baseline_f1": old,
            "qwen_f1": float(f1_score(targets[:, label], standalone[:, label],
                                      zero_division=0)),
            "candidate_f1": new, "delta": new - old,
            "qwen_selected_folds": len(nonzero), "accepted": accepted,
        })
        production.append({"label": name, "weight": weight, "ratio": ratio,
                           "accepted": accepted})

    payload = {
        "stage": "summarize-oof",
        "training_version": TRAINING_VERSION,
        "posts": len(frame), "folds": N_FOLDS,
        "baseline_macro_f1": float(f1_score(
            targets, baseline, average="macro", zero_division=0)),
        "qwen_standalone_macro_f1": float(f1_score(
            targets, standalone, average="macro", zero_division=0)),
        "nested_candidate_macro_f1": float(f1_score(
            targets, candidate, average="macro", zero_division=0)),
        "fold_results": [{k: v for k, v in row.items() if k != "selections"}
                         for row in selections],
        "per_label": per_label,
        "production_parameters": production,
    }
    (WORK / f"{MODEL_KEY}_oof_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8",
    )
    np.savez_compressed(
        WORK / f"{MODEL_KEY}_oof_probabilities.npz",
        probabilities=qwen.astype(np.float16), targets=targets.astype(np.int8),
        fold_membership=membership.astype(np.int8),
        row_id=frame.row_id.astype(str).to_numpy(dtype="U"),
    )
    print(json.dumps(payload, indent=2), flush=True)


def train_full(args):
    frame, targets, counts = load_train()
    test = pd.read_excel(locate("leaderboard.xlsx"))
    test["post"] = test["post"].fillna("").astype(str)
    model, head, tokenizer, input_device, output_device = load_model(targets.mean(0))
    combined_text = frame.post.tolist() + test.post.tolist()
    combined_targets = np.vstack([targets, np.zeros((len(test), len(FACTORS)),
                                                    dtype=np.float32)])
    combined_counts = np.vstack([counts, np.zeros((len(test), len(FACTORS)),
                                                  dtype=np.float32)])
    dataset = FactorPromptDataset(
        combined_text, combined_targets, combined_counts,
        tokenizer, PRESET["max_length"],
    )
    history = train(model, head, dataset, np.arange(len(frame)), targets,
                    input_device, output_device, args.epochs)
    test_indices = np.arange(len(frame), len(frame) + len(test))
    probability = infer(model, head, dataset, test_indices, input_device,
                        output_device, "Factor leaderboard inference")
    np.savez_compressed(
        WORK / f"{MODEL_KEY}_test_probabilities.npz",
        row_id=test.row_id.astype(str).to_numpy(dtype="U"),
        probabilities=probability.astype(np.float16),
        factors=np.asarray(FACTORS, dtype="U"),
    )
    payload = {
        "stage": "full", "training_version": TRAINING_VERSION,
        "model": MODEL_NAME, "train_posts": len(frame), "test_posts": len(test),
        "epochs": args.epochs, "history": history,
        "mean_probabilities": probability.mean(0).tolist(),
    }
    (WORK / f"{MODEL_KEY}_full_results.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8",
    )
    if args.save_adapter:
        adapter = WORK / f"{MODEL_KEY}_full_adapter"
        model.save_pretrained(adapter); tokenizer.save_pretrained(adapter)
        torch.save({"state_dict": {k: v.detach().cpu() for k, v in head.state_dict().items()},
                    "factors": FACTORS, "definitions": DEFINITIONS,
                    "prevalence": targets.mean(0)}, adapter / "factor_head.pt")
        print(f"Saved reusable full adapter and factor head: {adapter}", flush=True)
    print(json.dumps(payload, indent=2), flush=True)


def preflight():
    frame, targets, counts = load_train()
    split = np.load(locate("factor_baseline_oof.npz"), allow_pickle=True)
    membership = split["fold_membership"].astype(int)
    if not np.array_equal(split["targets"].astype(np.int8), targets.astype(np.int8)):
        raise RuntimeError("Packaged targets differ from train.xlsx")
    leakage = []
    groups = frame.anon_user_id.astype(str).to_numpy()
    for fold in range(N_FOLDS):
        fit = np.flatnonzero(membership != fold)
        valid = np.flatnonzero(membership == fold)
        leakage.append(len(set(groups[fit]) & set(groups[valid])))
    token = os.environ.get("HF_TOKEN") or None
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=token)
    prefix_tokens = len(tokenizer.encode(PREFIX, add_special_tokens=False))
    suffix_tokens = len(tokenizer.encode(SUFFIX, add_special_tokens=False))
    payload = {
        "training_version": TRAINING_VERSION,
        "preflight_passed": bool(not any(leakage)),
        "posts": len(frame), "users": int(len(set(groups))),
        "targets_shape": list(targets.shape),
        "duplicate_factor_occurrences": int((counts > 1).sum()),
        "fold_sizes": [int((membership == fold).sum()) for fold in range(N_FOLDS)],
        "user_overlap_by_fold": leakage,
        "support": {name: int(targets[:, label].sum())
                    for label, name in enumerate(FACTORS)},
        "prefix_tokens": prefix_tokens,
        "post_token_budget": PRESET["max_length"] - prefix_tokens - suffix_tokens,
    }
    print(json.dumps(payload, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["preflight", "fold", "summarize-oof", "full"],
                        default="fold")
    parser.add_argument("--fold", type=int, choices=range(N_FOLDS), default=0)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--save-adapter", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available() and args.stage not in ("preflight", "summarize-oof"):
        raise RuntimeError("Enable a Kaggle GPU accelerator before running")
    seed_everything(SEED + args.fold)
    print(json.dumps({
        "torch": torch.__version__, "cuda": torch.cuda.is_available(),
        "gpus": [torch.cuda.get_device_name(i)
                 for i in range(torch.cuda.device_count())],
        "arguments": vars(args),
    }, indent=2), flush=True)
    if args.stage == "preflight":
        preflight()
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
