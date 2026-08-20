"""DeBERTa-v3-large capacity gate for Task 1 (V46).

This is a deliberately bounded experiment inspired by the stronger-model
solutions from the 2024 BigData Cup.  It trains a partially frozen large
encoder on an inner user split, selects only the ensemble weight there, then
refits on the outer training users and evaluates once on untouched users.
Evidence remains the accepted V35 decoder output.
"""
from __future__ import annotations

import json
import math

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup

from analyze_task1_lexical_v11 import _lexical_experts, _softmax, _transformer_probability
from baseline import _post_phrase_f1
from configs.config import config
from inference.task1_evidence_v4 import correct_risk_only
from preprocess.preprocess import load_train_data
from trainer.task1_atomic_v25 import _load_records
from trainer.task1_risk_only_v27 import _v18_evidence
from utils.seed import seed_everything
from utils.task1_metric import task1_score


OUTPUT = config.OUTPUT_DIR / "task1_large_v46"
RESULTS = OUTPUT / "results.json"
INNER_CHECKPOINT = OUTPUT / "inner_trainable.pt"
OUTER_CHECKPOINT = OUTPUT / "outer_trainable.pt"
TRAINING_VERSION = "task1-deberta-v3-large-partial-v46"
MODEL_NAME = "microsoft/deberta-v3-large"
INNER_CAL_FOLD = 3
MAX_EPOCHS = 3
MAX_LENGTH = 512
ACCUMULATION = 8
TOP_TRAINABLE_LAYERS = 6


class HeadTailDataset(Dataset):
    """One 512-token view retaining both the opening and ending of a post."""

    def __init__(self, texts, labels):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
        rows = []
        content_length = MAX_LENGTH - tokenizer.num_special_tokens_to_add(pair=False)
        first = content_length // 2
        for text, label in tqdm(zip(texts, labels), total=len(texts),
                                desc="V46 head-tail tokenization"):
            tokens = tokenizer.encode(str(text), add_special_tokens=False)
            if len(tokens) > content_length:
                tokens = tokens[:first] + tokens[-(content_length - first):]
            # transformers>=5 exposes a slim DebertaV2Tokenizer without the
            # legacy prepare_for_model/build_inputs_with_special_tokens helpers.
            # DeBERTa's single-sequence layout is simply [CLS] text [SEP].
            input_ids = [tokenizer.cls_token_id, *tokens, tokenizer.sep_token_id]
            attention_mask = [1] * len(input_ids)
            padding = MAX_LENGTH - len(input_ids)
            input_ids.extend([tokenizer.pad_token_id] * padding)
            attention_mask.extend([0] * padding)
            rows.append((torch.tensor(input_ids, dtype=torch.long),
                         torch.tensor(attention_mask, dtype=torch.long),
                         int(label)))
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        ids, mask, label = self.rows[index]
        return {"input_ids": ids, "attention_mask": mask,
                "risk_labels": torch.tensor(label, dtype=torch.long)}


class LargeRiskModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(MODEL_NAME, dtype=torch.float16)
        self.encoder.config.use_cache = False
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False
        layers = self.encoder.encoder.layer
        for layer in layers[-TOP_TRAINABLE_LAYERS:]:
            # Keep FP32 master weights for trainable parameters.  Autocast will
            # still execute their matmuls in FP16, while GradScaler can safely
            # unscale the resulting FP32 gradients.
            layer.float()
            for parameter in layer.parameters():
                parameter.requires_grad = True
        hidden = int(self.encoder.config.hidden_size)
        self.projection = nn.Sequential(
            nn.LayerNorm(hidden * 2), nn.Linear(hidden * 2, hidden),
            nn.GELU(), nn.Dropout(.15),
        )
        self.classifier = nn.Linear(hidden, 4)
        self.ordinal = nn.Linear(hidden, 3)

    def forward(self, input_ids, attention_mask):
        hidden = self.encoder(input_ids=input_ids,
                              attention_mask=attention_mask).last_hidden_state
        weights = attention_mask.to(hidden.dtype).unsqueeze(-1)
        mean = (hidden * weights).sum(1) / weights.sum(1).clamp_min(1.0)
        # Masked max complements the global average with sparse high-risk cues.
        masked = hidden.masked_fill(~attention_mask.bool().unsqueeze(-1), -1e4)
        maximum = masked.max(1).values
        document = self.projection(torch.cat((mean, maximum), -1).float())
        return self.classifier(document), self.ordinal(document)


def _loader(dataset, indices, shuffle):
    return DataLoader(Subset(dataset, list(map(int, indices))), batch_size=1,
                      shuffle=shuffle, num_workers=0, pin_memory=True)


def _weights(labels, indices, device):
    selected = labels[np.asarray(indices)]
    counts = np.bincount(selected, minlength=4).astype(float)
    class_weights = np.sqrt(len(selected) / np.maximum(counts, 1.0))
    class_weights /= class_weights.mean()
    ordinal = selected[:, None] > np.arange(3)[None, :]
    positive = ordinal.sum(0).astype(float)
    ordinal_weights = np.sqrt((len(selected) - positive) / np.maximum(positive, 1.0))
    return (torch.tensor(class_weights, dtype=torch.float32, device=device),
            torch.tensor(ordinal_weights, dtype=torch.float32, device=device))


def _probability(logits, ordinal_logits):
    standard = torch.softmax(logits.float(), -1)
    cumulative = torch.sigmoid(ordinal_logits.float())
    cumulative = torch.cummin(cumulative, dim=1).values
    ordinal = torch.stack((
        1.0 - cumulative[:, 0], cumulative[:, 0] - cumulative[:, 1],
        cumulative[:, 1] - cumulative[:, 2], cumulative[:, 2],
    ), 1).clamp_min(1e-7)
    ordinal /= ordinal.sum(1, keepdim=True).clamp_min(1e-7)
    return (.8 * standard + .2 * ordinal).cpu().numpy()


@torch.no_grad()
def _infer(model, dataset, indices, device, description):
    model.eval(); rows = []
    for batch in tqdm(_loader(dataset, indices, False), desc=description):
        with torch.autocast(device_type="cuda", enabled=True):
            logits, ordinal = model(batch["input_ids"].to(device, non_blocking=True),
                                    batch["attention_mask"].to(device, non_blocking=True))
        rows.append(_probability(logits, ordinal)[0])
    return np.vstack(rows)


def _train(model, dataset, indices, labels, device, epochs, callback=None):
    loader = _loader(dataset, indices, True)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    head_ids = {id(parameter) for module in (
        model.projection, model.classifier, model.ordinal
    ) for parameter in module.parameters()}
    optimizer = AdamW([
        {"params": [p for p in trainable if id(p) not in head_ids], "lr": 5e-6},
        {"params": [p for p in trainable if id(p) in head_ids], "lr": 3e-5},
    ], weight_decay=.01)
    updates = math.ceil(len(loader) / ACCUMULATION) * epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, max(1, int(.08 * updates)), max(1, updates),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    class_weights, ordinal_weights = _weights(labels, indices, device)
    history = []
    for epoch in range(1, epochs + 1):
        model.train(); optimizer.zero_grad(set_to_none=True); losses = []
        progress = tqdm(loader, desc=f"V46 large epoch {epoch}/{epochs}")
        for step, batch in enumerate(progress, 1):
            target = batch["risk_labels"].to(device, non_blocking=True)
            ordinal_target = (target[:, None] >
                              torch.arange(3, device=device)[None, :]).float()
            with torch.autocast(device_type="cuda", enabled=True):
                logits, ordinal = model(
                    batch["input_ids"].to(device, non_blocking=True),
                    batch["attention_mask"].to(device, non_blocking=True),
                )
                ce = nn.functional.cross_entropy(
                    logits.float(), target, weight=class_weights,
                    label_smoothing=.03,
                )
                probability = torch.softmax(logits.float(), -1)
                focal = (((1.0 - probability.gather(1, target[:, None]).squeeze(1)) ** 2)
                         * nn.functional.cross_entropy(
                             logits.float(), target, reduction="none",
                         ) * class_weights[target]).mean()
                ordinal_loss = nn.functional.binary_cross_entropy_with_logits(
                    ordinal.float(), ordinal_target, pos_weight=ordinal_weights,
                )
                loss = (.70 * ce + .15 * focal + .30 * ordinal_loss) / ACCUMULATION
            scaler.scale(loss).backward()
            losses.append(float(loss.detach()) * ACCUMULATION)
            if step % ACCUMULATION == 0 or step == len(loader):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(trainable, 1.0)
                old_scale = scaler.get_scale(); scaler.step(optimizer); scaler.update()
                if scaler.get_scale() >= old_scale:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses))}
        history.append(row)
        print(f"V46 epoch={epoch} loss={row['train_loss']:.5f}", flush=True)
        if callback is not None:
            callback(epoch, model, row)
    return history


def _risk_probability(texts, transformer, large, lexical_decision, parameters, weight):
    neural = (1.0 - weight) * transformer + weight * large
    lexical = _softmax(lexical_decision, float(parameters["temperature"]))
    probability = ((1.0 - float(parameters["lexical_weight"])) * neural
                   + float(parameters["lexical_weight"]) * lexical)
    logits = np.log(np.clip(probability, 1e-8, 1.0))
    logits[:, 0] += float(parameters.get("indicator_bias", 0.0))
    logits[:, 2] += float(parameters.get("behavior_bias", 0.0))
    logits[:, 3] += float(parameters.get("attempt_bias", 0.0))
    logits -= logits.max(1, keepdims=True)
    probability = np.exp(logits); probability /= probability.sum(1, keepdims=True)
    prediction = probability.argmax(1)
    prediction = np.asarray([correct_risk_only(text, int(risk))
                             for text, risk in zip(texts, prediction)], dtype=np.int64)
    return prediction


def _save_trainable(model, path, selected, history):
    state = {name: value.detach().cpu() for name, value in model.state_dict().items()
             if name.startswith(("projection.", "classifier.", "ordinal."))
             or (name.startswith("encoder.encoder.layer.")
                 and int(name.split(".")[3]) >= 24 - TOP_TRAINABLE_LAYERS)}
    torch.save({"training_version": TRAINING_VERSION, "state_dict": state,
                "selected": selected, "history": history}, path)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("V46 requires CUDA")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    seed_everything(config.SEED + 4646)
    frame = load_train_data().reset_index(drop=True)
    labels = frame.risk_label.to_numpy(dtype=np.int64)
    groups = frame.anon_user_id.astype(str).to_numpy()
    outer_train, outer_valid = next(StratifiedGroupKFold(
        n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED,
    ).split(np.zeros(len(frame)), labels, groups))
    records, membership_map, outer_raw = _load_records()
    inner_fit = np.asarray([i for i in outer_train
                            if membership_map[int(i)] != INNER_CAL_FOLD])
    inner_cal = np.asarray([i for i in outer_train
                            if membership_map[int(i)] == INNER_CAL_FOLD])
    record_by_index = {int(row["global_index"]): row for row in records}
    inner_records = [record_by_index[int(i)] for i in inner_cal]
    inner_transformer = np.vstack([row["old_probability"] for row in inner_records])
    v36 = json.loads((config.OUTPUT_DIR / "task1_oof_risk_v36" / "calibration.json")
                     .read_text(encoding="utf-8"))
    inner_lexical = _lexical_experts(frame, inner_fit, inner_cal)[v36["expert"]]
    inner_texts = frame.text.iloc[inner_cal].astype(str).tolist()

    dataset = HeadTailDataset(frame.text.astype(str).tolist(), labels)
    device = torch.device("cuda")
    model = LargeRiskModel().to(device)
    selected = {"risk_f1": -1.0}; history = []

    def callback(epoch, current, row):
        nonlocal selected
        large = _infer(current, dataset, inner_cal, device,
                       f"V46 inner validation epoch {epoch}")
        candidates = []
        for weight in (0.0, .10, .20, .30, .40, .50):
            prediction = _risk_probability(
                inner_texts, inner_transformer, large, inner_lexical, v36, weight,
            )
            score = float(f1_score(labels[inner_cal], prediction,
                                   average="weighted", zero_division=0))
            candidates.append((score, weight))
        score, weight = max(candidates, key=lambda item: (item[0], -item[1]))
        row.update({"calibration_risk_f1": score, "large_weight": weight})
        print(f"V46 epoch={epoch} inner risk_f1={score:.6f} "
              f"weight={weight:.2f}", flush=True)
        if score > selected["risk_f1"]:
            selected = {"epoch": epoch, "risk_f1": score, "large_weight": weight}

    history = _train(model, dataset, inner_fit, labels, device, MAX_EPOCHS, callback)
    _save_trainable(model, INNER_CHECKPOINT, selected, history)
    del model
    torch.cuda.empty_cache()

    seed_everything(config.SEED + 4647)
    final = LargeRiskModel().to(device)
    refit_history = _train(final, dataset, outer_train, labels, device,
                           int(selected["epoch"]))
    _save_trainable(final, OUTER_CHECKPOINT, selected, refit_history)
    large_outer = _infer(final, dataset, outer_valid, device, "V46 untouched outer users")

    outer_records = outer_raw["records"]
    transformer_outer = _transformer_probability(
        __import__("datasets.dataset", fromlist=["SuicideRiskDataset"]).SuicideRiskDataset(
            config.CACHE_DIR / "train_cache.pt"), outer_valid, outer_records,
    )
    outer_lexical = _lexical_experts(frame, outer_train, outer_valid)[v36["expert"]]
    outer_texts = frame.text.iloc[outer_valid].astype(str).tolist()
    baseline = _risk_probability(
        outer_texts, transformer_outer, large_outer, outer_lexical, v36, 0.0,
    )
    candidate = _risk_probability(
        outer_texts, transformer_outer, large_outer, outer_lexical, v36,
        float(selected["large_weight"]),
    )

    v18 = json.loads((config.OUTPUT_DIR / "task1_candidate_v18" / "results.json")
                     .read_text(encoding="utf-8"))
    v35 = json.loads((config.OUTPUT_DIR / "task1_oof_decoder_v35" / "calibration.json")
                     .read_text(encoding="utf-8"))
    evidence_parameters = (v35["parameters_by_predicted_risk"]
                           if v35.get("adopted", False)
                           else v18["evidence_parameters_by_predicted_risk"])
    seed2 = torch.load(config.OUTPUT_DIR / "task1_seed_ensemble_v14" / "seed2_valid.pt",
                       map_location="cpu", weights_only=False)["rows"]
    baseline_evidence = _v18_evidence(outer_records, seed2, baseline, evidence_parameters)
    candidate_evidence = _v18_evidence(outer_records, seed2, candidate, evidence_parameters)
    gold = [list(frame.iloc[int(i)].evidence) for i in outer_valid]
    base_phrase = np.asarray([_post_phrase_f1(x, y)
                              for x, y in zip(baseline_evidence, gold)])
    new_phrase = np.asarray([_post_phrase_f1(x, y)
                             for x, y in zip(candidate_evidence, gold)])
    base_risk = float(f1_score(labels[outer_valid], baseline,
                               average="weighted", zero_division=0))
    new_risk = float(f1_score(labels[outer_valid], candidate,
                              average="weighted", zero_division=0))
    base_task = task1_score(base_risk, float(base_phrase.mean()))
    new_task = task1_score(new_risk, float(new_phrase.mean()))

    unique = np.unique(groups[outer_valid]); rng = np.random.default_rng(config.SEED + 4646)
    deltas = []
    for _ in range(4000):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        positions = np.concatenate([np.flatnonzero(groups[outer_valid] == user)
                                    for user in sampled])
        old_risk = f1_score(labels[outer_valid][positions], baseline[positions],
                            average="weighted", zero_division=0)
        next_risk = f1_score(labels[outer_valid][positions], candidate[positions],
                             average="weighted", zero_division=0)
        deltas.append(task1_score(next_risk, float(new_phrase[positions].mean()))
                      - task1_score(old_risk, float(base_phrase[positions].mean())))
    deltas = np.asarray(deltas)
    bootstrap = {"mean_delta": float(deltas.mean()),
                 "p05_delta": float(np.quantile(deltas, .05)),
                 "p95_delta": float(np.quantile(deltas, .95)),
                 "positive_fraction": float((deltas > 0).mean())}
    adopted = bool(new_task >= base_task + .003
                   and bootstrap["positive_fraction"] >= .80)
    payload = {"training_version": TRAINING_VERSION,
        "model": MODEL_NAME, "method": {"max_length": MAX_LENGTH,
            "head_tail": True, "trainable_top_layers": TOP_TRAINABLE_LAYERS,
            "loss": "0.70 CE + 0.15 focal + 0.30 CORAL-style ordinal"},
        "evaluation_scope": "inner user calibration; one untouched outer user fold",
        "selected": selected, "history": history, "refit_history": refit_history,
        "baseline_v36_v35": {"risk_f1": base_risk,
            "phrase_f1": float(base_phrase.mean()), "task1": base_task},
        "candidate": {"risk_f1": new_risk, "phrase_f1": float(new_phrase.mean()),
            "task1": new_task, "changed_predictions": int(np.sum(baseline != candidate)),
            "confusion": confusion_matrix(labels[outer_valid], candidate,
                                            labels=np.arange(4)).tolist()},
        "user_cluster_bootstrap": bootstrap, "adopted": adopted}
    RESULTS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    del final
    torch.cuda.empty_cache()
    return payload


if __name__ == "__main__":
    main()
