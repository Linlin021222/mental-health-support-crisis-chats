"""Kaggle 4-bit QLoRA capacity experiments for suicide-risk classification.

This is intentionally standalone: it only needs the files shipped beside it.
Run one strict user-disjoint fold first, send the JSON result back for review,
then run the remaining folds only for a promising model.
"""
from __future__ import annotations

import argparse
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
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Subset
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

LABELS = ["Indicator", "Ideation", "Behavior", "Attempt"]
LABEL_TO_ID = {label.casefold(): index for index, label in enumerate(LABELS)}
LETTERS = "ABCD"
SEED = 42

PRESETS = {
    "qwen3-8b": {
        "model": "Qwen/Qwen3-8B", "max_length": 1024, "rank": 16,
        "batch_size": 1, "accumulation": 16, "lr": 5e-5,
    },
    "qwen25-14b": {
        "model": "Qwen/Qwen2.5-14B-Instruct", "max_length": 768, "rank": 8,
        "batch_size": 1, "accumulation": 16, "lr": 4e-5,
    },
    "gemma2-9b": {
        "model": "google/gemma-2-9b-it", "max_length": 1024, "rank": 16,
        "batch_size": 1, "accumulation": 16, "lr": 5e-5,
    },
    "llama31-8b": {
        "model": "meta-llama/Llama-3.1-8B-Instruct", "max_length": 1024,
        "rank": 16, "batch_size": 1, "accumulation": 16, "lr": 5e-5,
    },
}

PREFIX = (
    "Classify the suicide risk expressed by the AUTHOR of the current Reddit post.\n"
    "A = Indicator: no explicit suicide mention by the author. Distress, "
    "hopelessness, anger, self-hatred, or another person's suicide alone remain A.\n"
    "B = Ideation: explicit suicidal thought, wish to die, or desire not to wake "
    "up, without a concrete plan or performed act.\n"
    "C = Behavior: explicit suicidal intent plus a plan, method, accessible means, "
    "preparation, or self-harm/suicidal behavior that is not an explicit attempt.\n"
    "D = Attempt: the author explicitly describes a past or recent suicide attempt "
    "or a performed potentially lethal act intended to die.\n"
    "Scope rules: distinguish the author from quoted/third-person speech; distinguish "
    "negation and hypothetical language; a genuine historical attempt is D.\n"
    "Reply with exactly one letter: A, B, C, or D.\n"
    "CURRENT POST:\n"
)
SUFFIX = "\nANSWER:"
SUICIDE_CUE = re.compile(
    r"suicid|kill myself|killing myself|end my life|want to die|wanna die|"
    r"wish i (?:was )?dead|better off dead|not wake up|overdos|slit|hang myself|"
    r"jump off|shoot myself|blow my (?:head|brains)", re.I,
)


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def locate(name):
    direct = ROOT / name
    if direct.exists():
        return direct
    candidates = list(Path("/kaggle/input").rglob(name)) if Path("/kaggle/input").exists() else []
    if len(candidates) != 1:
        raise FileNotFoundError(f"Could not uniquely locate {name}: {candidates}")
    return candidates[0]


def load_train():
    frame = pd.read_excel(locate("train.xlsx"))
    normalized = frame["suicide risk"].astype(str).str.strip().str.casefold()
    frame["risk_id"] = normalized.map(LABEL_TO_ID)
    if frame.risk_id.isna().any():
        raise ValueError("Unknown suicide-risk label in train.xlsx")
    frame["risk_id"] = frame.risk_id.astype(int)
    frame["post"] = frame["post"].fillna("").astype(str)
    frame["anon_user_id"] = frame["anon_user_id"].fillna(frame["row_id"]).astype(str)
    return frame


def _balanced_post_tokens(tokenizer, text, budget):
    tokens = tokenizer.encode(str(text), add_special_tokens=False)
    if len(tokens) <= budget:
        return tokens
    # Preserve beginning/end plus a cue-centred excerpt.  This is superior to
    # blind head-only truncation for long Reddit posts while remaining verbatim.
    cue_blocks = []
    for match in list(SUICIDE_CUE.finditer(str(text)))[:6]:
        cue_blocks.append(str(text)[max(0, match.start() - 180):match.end() + 240])
    cue_tokens = tokenizer.encode("\n[...]\n".join(cue_blocks), add_special_tokens=False)
    cue_budget = min(len(cue_tokens), int(.40 * budget))
    side = (budget - cue_budget) // 2
    result = tokens[:side]
    if cue_budget:
        result += cue_tokens[:cue_budget]
    result += tokens[-(budget - len(result)):]
    return result[:budget]


class PromptDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        prefix = tokenizer.encode(PREFIX, add_special_tokens=False)
        suffix = tokenizer.encode(SUFFIX, add_special_tokens=False)
        available = max_length - len(prefix) - len(suffix)
        if available < 128:
            raise ValueError("max_length is too small for the definition prompt")
        rows = []
        for text, label in tqdm(list(zip(texts, labels)), desc="Tokenizing posts"):
            post = _balanced_post_tokens(tokenizer, text, available)
            ids = prefix + post + suffix
            mask = [1] * len(ids)
            padding = max_length - len(ids)
            ids += [tokenizer.pad_token_id] * padding
            mask += [0] * padding
            rows.append((torch.tensor(ids), torch.tensor(mask), int(label)))
        self.rows = rows

    def __len__(self): return len(self.rows)

    def __getitem__(self, index):
        ids, mask, label = self.rows[index]
        return {"input_ids": ids, "attention_mask": mask,
                "label": torch.tensor(label, dtype=torch.long)}


def _single_token_verbalizers(tokenizer):
    candidates = [[f" {letter}", letter] for letter in LETTERS]
    ids = []
    for choices in candidates:
        selected = None
        for choice in choices:
            encoded = tokenizer.encode(choice, add_special_tokens=False)
            if len(encoded) == 1:
                selected = encoded[0]; break
        if selected is None:
            raise RuntimeError(f"No single-token verbalizer for {choices}")
        ids.append(selected)
    return ids


def load_model(preset):
    token = os.environ.get("HF_TOKEN") or None
    tokenizer = AutoTokenizer.from_pretrained(preset["model"], token=token)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    label_ids = _single_token_verbalizers(tokenizer)
    quantization = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
    )
    max_memory = {index: "14500MiB" for index in range(torch.cuda.device_count())}
    max_memory["cpu"] = "26GiB"
    model = AutoModelForCausalLM.from_pretrained(
        preset["model"], token=token, quantization_config=quantization,
        device_map="auto", max_memory=max_memory, torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=preset["rank"],
        lora_alpha=2 * preset["rank"], lora_dropout=.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    ))
    model.print_trainable_parameters()
    input_device = model.get_input_embeddings().weight.device
    label_ids = torch.tensor(label_ids, dtype=torch.long,
                             device=model.get_output_embeddings().weight.device)
    return model, tokenizer, label_ids, input_device


def verbalizer_logits(model, input_ids, attention_mask, label_ids):
    base = model.get_base_model()
    hidden = base.model(input_ids=input_ids, attention_mask=attention_mask,
                        use_cache=False).last_hidden_state
    positions = (attention_mask.sum(1) - 1).to(hidden.device)
    row = torch.arange(hidden.size(0), device=hidden.device)
    final = hidden[row, positions]
    weights = base.get_output_embeddings().weight.index_select(0, label_ids)
    return F.linear(final.to(weights.device).float(), weights.float())


def macro_double_soft_f1(logits, labels):
    target = F.one_hot(labels, num_classes=4).float()
    probability = torch.sigmoid(logits.float())
    tp = (probability * target).sum(0)
    fp = (probability * (1. - target)).sum(0)
    fn = ((1. - probability) * target).sum(0)
    tn = ((1. - probability) * (1. - target)).sum(0)
    positive = 2. * tp / (2. * tp + fn + fp + 1e-7)
    negative = 2. * tn / (2. * tn + fn + fp + 1e-7)
    return (1. - .5 * (positive + negative)).mean()


def loader(dataset, indices, batch_size, shuffle):
    return DataLoader(Subset(dataset, list(map(int, indices))),
                      batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=True)


@torch.no_grad()
def infer(model, dataset, indices, preset, label_ids, input_device, description):
    model.eval(); probabilities = []
    for batch in tqdm(loader(dataset, indices, preset["batch_size"], False),
                      desc=description):
        with torch.autocast("cuda", dtype=torch.float16):
            logits = verbalizer_logits(
                model, batch["input_ids"].to(input_device, non_blocking=True),
                batch["attention_mask"].to(input_device, non_blocking=True), label_ids,
            )
        probabilities.append(torch.softmax(logits, -1).cpu().numpy())
    return np.vstack(probabilities)


def train(model, dataset, indices, labels, preset, label_ids, input_device, epochs):
    batches = loader(dataset, indices, preset["batch_size"], True)
    counts = np.bincount(labels[np.asarray(indices)], minlength=4)
    weights = np.sqrt(len(indices) / np.maximum(counts, 1))
    weights = torch.tensor(weights / weights.mean(), dtype=torch.float32,
                           device=label_ids.device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = AdamW(parameters, lr=preset["lr"], weight_decay=.1)
    updates = math.ceil(len(batches) / preset["accumulation"]) * epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, max(1, int(.06 * updates)), max(1, updates))
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    history = []
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(1, epochs + 1):
        model.train(); losses = []
        progress = tqdm(batches, desc=f"QLoRA epoch {epoch}/{epochs}")
        for step, batch in enumerate(progress, 1):
            target = batch["label"].to(label_ids.device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = verbalizer_logits(
                    model, batch["input_ids"].to(input_device, non_blocking=True),
                    batch["attention_mask"].to(input_device, non_blocking=True), label_ids,
                )
                ce = F.cross_entropy(logits, target, weight=weights,
                                     label_smoothing=.03)
                soft_f1 = macro_double_soft_f1(logits, target)
                loss = (.65 * ce + .35 * soft_f1) / preset["accumulation"]
            scaler.scale(loss).backward()
            losses.append(float(loss.detach()) * preset["accumulation"])
            if step % preset["accumulation"] == 0 or step == len(batches):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(parameters, 1.)
                previous = scaler.get_scale(); scaler.step(optimizer); scaler.update()
                if scaler.get_scale() >= previous:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            progress.set_postfix(loss=f"{np.mean(losses[-50:]):.4f}")
        history.append({"epoch": epoch, "loss": float(np.mean(losses))})
        print(json.dumps(history[-1]), flush=True)
    return history


def _metrics(truth, probability):
    prediction = probability.argmax(1)
    return {
        "weighted_f1": float(f1_score(truth, prediction, average="weighted", zero_division=0)),
        "macro_f1": float(f1_score(truth, prediction, average="macro", zero_division=0)),
        "confusion": confusion_matrix(truth, prediction, labels=np.arange(4)).tolist(),
        "report": classification_report(truth, prediction, target_names=LABELS,
                                         output_dict=True, zero_division=0),
    }


def run_fold(args, preset):
    frame = load_train(); labels = frame.risk_id.to_numpy()
    split = np.load(locate("baseline_oof.npz"))
    global_indices = split["global_indices"].astype(int)
    membership = split["fold_membership"].astype(int)
    fit = global_indices[membership != args.fold]
    valid = global_indices[membership == args.fold]
    if set(frame.anon_user_id.iloc[fit]) & set(frame.anon_user_id.iloc[valid]):
        raise RuntimeError("User leakage detected in packaged split")
    print(f"MODEL={preset['model']} fold={args.fold} train={len(fit)} valid={len(valid)}", flush=True)
    model, tokenizer, label_ids, input_device = load_model(preset)
    dataset = PromptDataset(frame.post.tolist(), labels, tokenizer, preset["max_length"])
    torch.cuda.reset_peak_memory_stats()
    history = train(model, dataset, fit, labels, preset, label_ids, input_device, args.epochs)
    probability = infer(model, dataset, valid, preset, label_ids, input_device,
                        f"Fold {args.fold} inference")
    metric = _metrics(labels[valid], probability)
    peak = float(torch.cuda.max_memory_allocated() / 1024 ** 3)
    payload = {
        "stage": "fold", "model_key": args.model, "model": preset["model"],
        "fold": args.fold, "train_posts": len(fit), "valid_posts": len(valid),
        "user_disjoint": True, "epochs": args.epochs,
        "max_length": preset["max_length"], "history": history,
        "standalone": metric, "peak_allocated_gib": peak,
    }
    stem = f"{args.model}_fold{args.fold}"
    np.savez_compressed(WORK / f"{stem}_probabilities.npz",
                        global_indices=valid, probabilities=probability,
                        truth=labels[valid])
    (WORK / f"{stem}_results.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")
    if args.save_adapter:
        adapter = WORK / f"{stem}_adapter"
        model.save_pretrained(adapter); tokenizer.save_pretrained(adapter)
    print(json.dumps(payload, indent=2), flush=True)


def summarize_oof(args):
    frame = load_train(); labels = frame.risk_id.to_numpy()
    split = np.load(locate("baseline_oof.npz"))
    global_indices = split["global_indices"].astype(int)
    membership = split["fold_membership"].astype(int)
    baseline = split["probabilities"].astype(np.float64)
    position = {int(index): place for place, index in enumerate(global_indices)}
    llm = np.zeros_like(baseline)
    for fold in range(4):
        path = WORK / f"{args.model}_fold{fold}_probabilities.npz"
        if not path.exists():
            path = locate(path.name)
        saved = np.load(path)
        locations = [position[int(index)] for index in saved["global_indices"]]
        llm[locations] = saved["probabilities"]
    truth = labels[global_indices]
    crossfit = np.zeros(len(truth), dtype=int); selections = []
    for fold in range(4):
        fit = np.flatnonzero(membership != fold); held = np.flatnonzero(membership == fold)
        rows = []
        for temperature in (.7, 1., 1.3):
            scaled = np.log(np.clip(llm, 1e-8, 1.)) / temperature
            scaled = np.exp(scaled - scaled.max(1, keepdims=True)); scaled /= scaled.sum(1, keepdims=True)
            for weight in (0., .10, .20, .30, .40, .50, .70, 1.):
                probability = (1. - weight) * baseline + weight * scaled
                score = f1_score(truth[fit], probability[fit].argmax(1),
                                 average="weighted", zero_division=0)
                rows.append((score, -weight, temperature, weight, probability))
        _, _, temperature, weight, probability = max(rows, key=lambda row: row[:2])
        crossfit[held] = probability[held].argmax(1)
        selections.append({"fold": fold, "temperature": temperature, "weight": weight,
                           "heldout_f1": float(f1_score(
                               truth[held], crossfit[held], average="weighted", zero_division=0))})
    baseline_metric = _metrics(truth, baseline)
    llm_metric = _metrics(truth, llm)
    crossfit_metric = {
        "weighted_f1": float(f1_score(truth, crossfit, average="weighted", zero_division=0)),
        "macro_f1": float(f1_score(truth, crossfit, average="macro", zero_division=0)),
        "confusion": confusion_matrix(truth, crossfit, labels=np.arange(4)).tolist(),
    }
    payload = {"stage": "summarize-oof", "model_key": args.model,
               "posts": len(truth), "baseline_neural": baseline_metric,
               "llm_standalone": llm_metric, "nested_blend": crossfit_metric,
               "fold_selections": selections}
    (WORK / f"{args.model}_oof_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


def train_full(args, preset):
    frame = load_train(); labels = frame.risk_id.to_numpy()
    test = pd.read_excel(locate("leaderboard.xlsx"))
    test["post"] = test.post.fillna("").astype(str)
    model, tokenizer, label_ids, input_device = load_model(preset)
    combined_text = frame.post.tolist() + test.post.tolist()
    combined_labels = labels.tolist() + [0] * len(test)
    dataset = PromptDataset(combined_text, combined_labels, tokenizer, preset["max_length"])
    history = train(model, dataset, np.arange(len(frame)), labels, preset,
                    label_ids, input_device, args.epochs)
    test_indices = np.arange(len(frame), len(frame) + len(test))
    probability = infer(model, dataset, test_indices, preset, label_ids, input_device,
                        "Leaderboard inference")
    np.savez_compressed(WORK / f"{args.model}_test_probabilities.npz",
                        row_id=test.row_id.astype(str).to_numpy(), probabilities=probability)
    baseline = pd.read_csv(locate("baseline_panda.csv"))
    if baseline.row_id.astype(str).tolist() != test.row_id.astype(str).tolist():
        raise RuntimeError("baseline_panda.csv row order differs from leaderboard.xlsx")
    # Always save standalone and conservative candidates.  Do not submit all
    # of them: use OOF results to choose one after reporting them to Codex.
    for threshold in (.00, .70, .80):
        result = baseline.copy()
        prediction = probability.argmax(1)
        if threshold > 0:
            old = result.risk_level.str.casefold().map(LABEL_TO_ID).to_numpy()
            change = probability.max(1) >= threshold
            prediction = np.where(change, prediction, old)
        result["risk_level"] = [LABELS[int(value)] for value in prediction]
        suffix = "standalone" if threshold == 0 else f"conf{int(100*threshold)}"
        result.to_csv(WORK / f"panda_{args.model}_{suffix}.csv", index=False)
    payload = {"stage": "full", "model_key": args.model, "model": preset["model"],
               "train_posts": len(frame), "test_posts": len(test),
               "epochs": args.epochs, "history": history,
               "prediction_counts": {LABELS[i]: int((probability.argmax(1) == i).sum())
                                     for i in range(4)},
               "mean_confidence": float(probability.max(1).mean())}
    (WORK / f"{args.model}_full_results.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["fold", "summarize-oof", "full"], default="fold")
    parser.add_argument("--model", choices=sorted(PRESETS), default="qwen3-8b")
    parser.add_argument("--fold", type=int, choices=range(4), default=0)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--save-adapter", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available() and args.stage != "summarize-oof":
        raise RuntimeError("Enable a Kaggle GPU accelerator before running")
    seed_everything(SEED + args.fold)
    print(json.dumps({"torch": torch.__version__, "cuda": torch.cuda.is_available(),
                      "gpus": [torch.cuda.get_device_name(i)
                               for i in range(torch.cuda.device_count())],
                      "arguments": vars(args)}, indent=2), flush=True)
    if args.stage == "fold":
        run_fold(args, PRESETS[args.model])
    elif args.stage == "summarize-oof":
        summarize_oof(args)
    else:
        train_full(args, PRESETS[args.model])
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
