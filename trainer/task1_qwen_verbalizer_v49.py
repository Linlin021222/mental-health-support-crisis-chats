"""Strict one-fold Qwen verbalizer gate for Task 1 (V49).

Unlike V47/V48, this experiment does not create a randomly initialized
classification head.  Four existing language-model vocabulary rows (A-D)
act as label prototypes, while QLoRA adapts attention projections.
"""
from __future__ import annotations

import json
import math

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import get_cosine_schedule_with_warmup

from analyze_task1_lexical_v11 import _lexical_experts, _transformer_probability
from baseline import _post_phrase_f1
from configs.config import config
from preprocess.preprocess import load_train_data
from trainer.task1_atomic_v25 import _load_records
from trainer.task1_large_v46 import _risk_probability
from trainer.task1_risk_only_v27 import _v18_evidence
from utils.seed import seed_everything
from utils.task1_metric import task1_score


OUTPUT = config.OUTPUT_DIR / "task1_qwen_verbalizer_v49"
RESULTS = OUTPUT / "results.json"
ADAPTER = OUTPUT / "strict_adapter"
TRAINING_VERSION = "task1-qwen25-3b-verbalizer-v49"
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
MAX_LENGTH = 256
BATCH_SIZE = 2
ACCUMULATION = 8
EPOCHS = 1
USE_GRADIENT_CHECKPOINTING = False
PREDECLARED_WEIGHT = .20
PREFIX = (
    "Classify the Reddit post. Reply with exactly one letter.\n"
    "A = Indicator: no explicit suicide mention.\n"
    "B = Ideation: explicit wish to die or suicidal thought, but no plan.\n"
    "C = Behavior: suicidal expression plus a plan, method, means, or self-harm behavior.\n"
    "D = Attempt: a past or recent suicide attempt is explicitly described.\n"
    "Post:\n"
)
SUFFIX = "\nAnswer:"


class PromptDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        prefix = tokenizer.encode(PREFIX, add_special_tokens=False)
        suffix = tokenizer.encode(SUFFIX, add_special_tokens=False)
        available = MAX_LENGTH - len(prefix) - len(suffix)
        first = available // 2
        self.rows = []
        for text, label in tqdm(zip(texts, labels), total=len(texts),
                                desc="V49 verbalizer tokenization"):
            post = tokenizer.encode(str(text), add_special_tokens=False)
            if len(post) > available:
                post = post[:first] + post[-(available - first):]
            ids = prefix + post + suffix
            mask = [1] * len(ids)
            padding = MAX_LENGTH - len(ids)
            ids += [tokenizer.pad_token_id] * padding
            mask += [0] * padding
            self.rows.append((torch.tensor(ids, dtype=torch.long),
                              torch.tensor(mask, dtype=torch.long), int(label)))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        ids, mask, label = self.rows[index]
        return {"input_ids": ids, "attention_mask": mask,
                "risk_labels": torch.tensor(label, dtype=torch.long)}


def _model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    label_tokens = [tokenizer.encode(" " + letter, add_special_tokens=False)
                    for letter in "ABCD"]
    if any(len(tokens) != 1 for tokens in label_tokens):
        raise RuntimeError(f"V49 requires single-token verbalizers: {label_tokens}")
    label_token_ids = torch.tensor([tokens[0] for tokens in label_tokens],
                                   dtype=torch.long, device="cuda")
    quantization = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=quantization, device_map={"": 0},
        local_files_only=True,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=USE_GRADIENT_CHECKPOINTING,
    )
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16, lora_dropout=.05,
        target_modules=["q_proj", "v_proj"],
    ))
    return model, tokenizer, label_token_ids


def _loader(dataset, indices, shuffle):
    return DataLoader(Subset(dataset, list(map(int, indices))),
                      batch_size=BATCH_SIZE, shuffle=shuffle,
                      num_workers=0, pin_memory=True)


def _verbalizer_logits(model, input_ids, attention_mask, label_token_ids):
    """Project only the final prompt state onto four pretrained LM-head rows."""
    causal_lm = model.get_base_model()
    hidden = causal_lm.model(input_ids=input_ids,
                             attention_mask=attention_mask,
                             use_cache=False).last_hidden_state
    positions = attention_mask.sum(1) - 1
    final_hidden = hidden[torch.arange(hidden.size(0), device=hidden.device), positions]
    label_weights = causal_lm.lm_head.weight.index_select(0, label_token_ids)
    return F.linear(final_hidden.float(), label_weights.float())


@torch.no_grad()
def _infer(model, dataset, indices, label_token_ids, description):
    model.eval(); rows = []
    for batch in tqdm(_loader(dataset, indices, False), desc=description):
        with torch.autocast(device_type="cuda", enabled=True):
            logits = _verbalizer_logits(
                model, batch["input_ids"].cuda(non_blocking=True),
                batch["attention_mask"].cuda(non_blocking=True), label_token_ids)
        rows.append(torch.softmax(logits.float(), -1).cpu().numpy())
    return np.vstack(rows)


def _train(model, dataset, indices, labels, label_token_ids):
    loader = _loader(dataset, indices, True)
    counts = np.bincount(labels[np.asarray(indices)], minlength=4).astype(float)
    weights = np.sqrt(len(indices) / np.maximum(counts, 1.0))
    weights = torch.tensor(weights / weights.mean(), dtype=torch.float32, device="cuda")
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable, lr=1e-4, weight_decay=.01)
    updates = math.ceil(len(loader) / ACCUMULATION) * EPOCHS
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, max(1, int(.06 * updates)), max(1, updates))
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    history = []
    for epoch in range(1, EPOCHS + 1):
        model.train(); optimizer.zero_grad(set_to_none=True); losses = []
        for step, batch in enumerate(tqdm(loader, desc=f"V49 QLoRA epoch {epoch}"), 1):
            target = batch["risk_labels"].cuda(non_blocking=True)
            with torch.autocast(device_type="cuda", enabled=True):
                logits = _verbalizer_logits(
                    model, batch["input_ids"].cuda(non_blocking=True),
                    batch["attention_mask"].cuda(non_blocking=True), label_token_ids)
                loss = F.cross_entropy(logits, target, weight=weights,
                                       label_smoothing=.03) / ACCUMULATION
            scaler.scale(loss).backward()
            losses.append(float(loss.detach()) * ACCUMULATION)
            if step % ACCUMULATION == 0 or step == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                old_scale = scaler.get_scale(); scaler.step(optimizer); scaler.update()
                if scaler.get_scale() >= old_scale:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses))})
        print(f"V49 epoch={epoch} loss={history[-1]['train_loss']:.5f}", flush=True)
    return history


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("V49 requires CUDA")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    seed_everything(config.SEED + 4949)
    frame = load_train_data().reset_index(drop=True)
    labels = frame.risk_label.to_numpy(dtype=np.int64)
    groups = frame.anon_user_id.astype(str).to_numpy()
    train_idx, valid_idx = next(StratifiedGroupKFold(
        n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED,
    ).split(np.zeros(len(frame)), labels, groups))
    model, tokenizer, label_token_ids = _model_and_tokenizer()
    dataset = PromptDataset(frame.text.astype(str).tolist(), labels, tokenizer)
    history = _train(model, dataset, train_idx, labels, label_token_ids)
    qwen_probability = _infer(model, dataset, valid_idx, label_token_ids,
                              "V49 untouched outer users")
    model.save_pretrained(ADAPTER); tokenizer.save_pretrained(ADAPTER)

    _, _, outer_raw = _load_records(); outer_records = outer_raw["records"]
    from datasets.dataset import SuicideRiskDataset
    transformer = _transformer_probability(
        SuicideRiskDataset(config.CACHE_DIR / "train_cache.pt"), valid_idx, outer_records)
    v36 = json.loads((config.OUTPUT_DIR / "task1_oof_risk_v36" / "calibration.json")
                     .read_text(encoding="utf-8"))
    lexical = _lexical_experts(frame, train_idx, valid_idx)[v36["expert"]]
    texts = frame.text.iloc[valid_idx].astype(str).tolist()
    baseline = _risk_probability(texts, transformer, qwen_probability, lexical, v36, 0.0)
    candidate = _risk_probability(texts, transformer, qwen_probability, lexical,
                                  v36, PREDECLARED_WEIGHT)

    v18 = json.loads((config.OUTPUT_DIR / "task1_candidate_v18" / "results.json")
                     .read_text(encoding="utf-8"))
    v35 = json.loads((config.OUTPUT_DIR / "task1_oof_decoder_v35" / "calibration.json")
                     .read_text(encoding="utf-8"))
    evidence_parameters = (v35["parameters_by_predicted_risk"]
                           if v35.get("adopted", False)
                           else v18["evidence_parameters_by_predicted_risk"])
    seed2 = torch.load(config.OUTPUT_DIR / "task1_seed_ensemble_v14" / "seed2_valid.pt",
                       map_location="cpu", weights_only=False)["rows"]
    old_evidence = _v18_evidence(outer_records, seed2, baseline, evidence_parameters)
    new_evidence = _v18_evidence(outer_records, seed2, candidate, evidence_parameters)
    gold = [list(frame.iloc[int(i)].evidence) for i in valid_idx]
    old_phrase = np.asarray([_post_phrase_f1(x, y) for x, y in zip(old_evidence, gold)])
    new_phrase = np.asarray([_post_phrase_f1(x, y) for x, y in zip(new_evidence, gold)])

    def metrics(prediction, phrase):
        risk = float(f1_score(labels[valid_idx], prediction,
                              average="weighted", zero_division=0))
        return {"risk_f1": risk, "phrase_f1": float(phrase.mean()),
                "task1": task1_score(risk, float(phrase.mean()))}

    base = metrics(baseline, old_phrase); fixed = metrics(candidate, new_phrase)
    standalone = {"risk_f1": float(f1_score(labels[valid_idx],
                   qwen_probability.argmax(1), average="weighted", zero_division=0))}
    unique = np.unique(groups[valid_idx]); rng = np.random.default_rng(config.SEED + 4949)
    deltas = []
    for _ in range(4000):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        positions = np.concatenate([np.flatnonzero(groups[valid_idx] == user)
                                    for user in sampled])
        old_risk = f1_score(labels[valid_idx][positions], baseline[positions],
                            average="weighted", zero_division=0)
        new_risk = f1_score(labels[valid_idx][positions], candidate[positions],
                            average="weighted", zero_division=0)
        deltas.append(task1_score(new_risk, float(new_phrase[positions].mean()))
                      - task1_score(old_risk, float(old_phrase[positions].mean())))
    deltas = np.asarray(deltas)
    bootstrap = {"mean_delta": float(deltas.mean()),
                 "p05_delta": float(np.quantile(deltas, .05)),
                 "p95_delta": float(np.quantile(deltas, .95)),
                 "positive_fraction": float((deltas > 0).mean())}
    promising = bool(fixed["task1"] >= base["task1"] + .003
                     and bootstrap["positive_fraction"] >= .80)
    payload = {"training_version": TRAINING_VERSION, "model": MODEL_NAME,
        "method": {"head": "pretrained LM verbalizer A/B/C/D", "quantization": "NF4 4-bit",
                   "lora_rank": 8, "targets": ["q_proj", "v_proj"],
                   "max_length": MAX_LENGTH, "epochs": EPOCHS,
                   "predeclared_weight": PREDECLARED_WEIGHT},
        "evaluation_scope": "one untouched outer user fold; stage-one capacity gate",
        "history": history, "standalone": standalone,
        "baseline_v36_v35": base,
        "fixed_candidate": {**fixed,
            "changed_predictions": int(np.sum(baseline != candidate)),
            "confusion": confusion_matrix(labels[valid_idx], candidate,
                                            labels=np.arange(4)).tolist()},
        "user_cluster_bootstrap": bootstrap,
        "promising_for_full_oof": promising, "adopted": False}
    RESULTS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    return payload


if __name__ == "__main__":
    main()
