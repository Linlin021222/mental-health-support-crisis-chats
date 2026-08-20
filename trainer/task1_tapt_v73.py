"""Strict outer-holdout screen of task-adaptive pretraining for Task 1.

The DeBERTa encoder first performs parameter-efficient masked-language-model
training on *unlabelled outer-training Reddit posts only*.  Frozen document
embeddings before and after TAPT are then compared with the same supervised
linear probe.  All decoding choices are selected on an inner user-disjoint
calibration fold; the 330-post outer user holdout is evaluated exactly once.
"""
from __future__ import annotations

import json
import math

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.optim import AdamW
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup
from transformers.activations import ACT2FN
from transformers.utils.hub import cached_file

from baseline import _post_phrase_f1
from configs.config import config
from inference.task1_evidence_v4 import apply_evidence_policy, decode_model_evidence
from inference.task1_polarity_v63 import polarity_candidate
from preprocess.preprocess import load_train_data
from trainer.task1_atomic_v25 import _load_records
from utils.seed import seed_everything
from utils.task1_metric import task1_score


OUTPUT = config.OUTPUT_DIR / "task1_tapt_v73"
RESULTS = OUTPUT / "results.json"
CHECKPOINT = OUTPUT / "tapt_trainable.pt"
EMBEDDINGS = OUTPUT / "embeddings.npz"
TRAINING_VERSION = "task1-parameter-efficient-tapt-v73"
MODEL_NAME = config.MODEL_NAME
MAX_LENGTH = 256
MLM_BATCH_SIZE = 2
MLM_ACCUMULATION = 8
MLM_EPOCHS = 1
TOP_ADAPTED_LAYERS = 4
INNER_CAL_FOLD = 3


class OriginalDebertaMLM(nn.Module):
    """DeBERTa-v3 MLM using the checkpoint's original (pre-v5) head names."""

    def __init__(self):
        super().__init__()
        self.deberta = AutoModel.from_pretrained(
            MODEL_NAME, local_files_only=True, dtype=torch.float32)
        hidden = int(self.deberta.config.hidden_size)
        self.lm_dense = nn.Linear(hidden, hidden)
        self.lm_norm = nn.LayerNorm(hidden, eps=float(self.deberta.config.layer_norm_eps))
        self.lm_bias = nn.Parameter(torch.zeros(int(self.deberta.config.vocab_size)))
        path = cached_file(MODEL_NAME, "pytorch_model.bin", local_files_only=True)
        state = torch.load(path, map_location="cpu", weights_only=True)
        self.lm_dense.load_state_dict({
            "weight": state["lm_predictions.lm_head.dense.weight"],
            "bias": state["lm_predictions.lm_head.dense.bias"],
        })
        self.lm_norm.load_state_dict({
            "weight": state["lm_predictions.lm_head.LayerNorm.weight"],
            "bias": state["lm_predictions.lm_head.LayerNorm.bias"],
        })
        self.lm_bias.data.copy_(state["lm_predictions.lm_head.bias"])
        self.activation = ACT2FN[str(self.deberta.config.hidden_act)]

    def forward(self, input_ids, attention_mask, labels):
        hidden = self.deberta(input_ids=input_ids,
                              attention_mask=attention_mask).last_hidden_state
        hidden = self.lm_norm(self.activation(self.lm_dense(hidden)))
        # DeBERTa-v3's MLM decoder is tied to the input word embeddings.
        logits = F.linear(hidden, self.deberta.embeddings.word_embeddings.weight,
                          self.lm_bias)
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                               labels.reshape(-1), ignore_index=-100)


class HeadTailDataset(Dataset):
    def __init__(self, texts, tokenizer):
        rows = []
        content = MAX_LENGTH - 2; first = content // 2
        for text in tqdm(texts, desc="V73 head-tail tokenization"):
            tokens = tokenizer.encode(str(text), add_special_tokens=False)
            if len(tokens) > content:
                tokens = tokens[:first] + tokens[-(content - first):]
            ids = [tokenizer.cls_token_id, *tokens, tokenizer.sep_token_id]
            mask = [1] * len(ids)
            pad = MAX_LENGTH - len(ids)
            ids += [tokenizer.pad_token_id] * pad
            mask += [0] * pad
            rows.append((torch.tensor(ids, dtype=torch.long),
                         torch.tensor(mask, dtype=torch.long)))
        self.rows = rows

    def __len__(self): return len(self.rows)

    def __getitem__(self, index):
        ids, mask = self.rows[index]
        return {"input_ids": ids, "attention_mask": mask}


def _mask_batch(input_ids, attention_mask, tokenizer):
    labels = input_ids.clone()
    probability = torch.full(labels.shape, .15, device=labels.device)
    special = (
        (input_ids == tokenizer.cls_token_id)
        | (input_ids == tokenizer.sep_token_id)
        | (input_ids == tokenizer.pad_token_id)
        | ~attention_mask.bool()
    )
    probability.masked_fill_(special, 0.0)
    masked = torch.bernoulli(probability).bool()
    # Avoid an empty MLM target for very short posts.
    for row in range(masked.shape[0]):
        if not masked[row].any():
            candidates = torch.nonzero(~special[row], as_tuple=False).flatten()
            if len(candidates):
                masked[row, candidates[len(candidates) // 2]] = True
    labels[~masked] = -100
    replace = torch.bernoulli(torch.full(labels.shape, .8, device=labels.device)).bool() & masked
    input_ids = input_ids.clone(); input_ids[replace] = tokenizer.mask_token_id
    random_mask = (torch.bernoulli(torch.full(labels.shape, .5, device=labels.device)).bool()
                   & masked & ~replace)
    random_words = torch.randint(len(tokenizer), labels.shape, device=labels.device)
    input_ids[random_mask] = random_words[random_mask]
    return input_ids, labels


def _configure_tapt(model):
    for parameter in model.parameters():
        parameter.requires_grad = False
    layers = model.deberta.encoder.layer
    for layer in layers[-TOP_ADAPTED_LAYERS:]:
        for parameter in layer.parameters():
            parameter.requires_grad = True
    # The 128k-token MLM decoder is intentionally frozen.  Its fixed language
    # space still supplies the MLM gradient to the encoder, while omitting its
    # optimizer states keeps the experiment within an 8 GB laptop GPU.
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def _encode(model, dataset, indices, device, description):
    loader = DataLoader(Subset(dataset, list(map(int, indices))), batch_size=8,
                        shuffle=False, num_workers=0, pin_memory=True)
    encoder = model.deberta; encoder.eval(); rows = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=description):
            ids = batch["input_ids"].to(device, non_blocking=True)
            mask = batch["attention_mask"].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                hidden = encoder(input_ids=ids, attention_mask=mask).last_hidden_state
            weights = mask.to(hidden.dtype).unsqueeze(-1)
            mean = (hidden * weights).sum(1) / weights.sum(1).clamp_min(1.0)
            maximum = hidden.masked_fill(~mask.bool().unsqueeze(-1), -1e4).max(1).values
            rows.append(torch.cat((hidden[:, 0], mean, maximum), 1).float().cpu().numpy())
    return np.vstack(rows).astype(np.float32)


def _train_tapt(model, dataset, indices, tokenizer, device):
    trainable = _configure_tapt(model)
    loader = DataLoader(Subset(dataset, list(map(int, indices))),
                        batch_size=MLM_BATCH_SIZE, shuffle=True, num_workers=0,
                        pin_memory=True)
    optimizer = AdamW(trainable, lr=1.5e-5, weight_decay=.01)
    updates = math.ceil(len(loader) / MLM_ACCUMULATION) * MLM_EPOCHS
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, max(1, int(.06 * updates)), max(1, updates))
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    history = []
    for epoch in range(1, MLM_EPOCHS + 1):
        model.train(); optimizer.zero_grad(set_to_none=True); losses = []
        progress = tqdm(loader, desc=f"V73 unsupervised MLM {epoch}/{MLM_EPOCHS}")
        for step, batch in enumerate(progress, 1):
            mask = batch["attention_mask"].to(device, non_blocking=True)
            ids = batch["input_ids"].to(device, non_blocking=True)
            masked_ids, labels = _mask_batch(ids, mask, tokenizer)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss = model(input_ids=masked_ids, attention_mask=mask,
                             labels=labels) / MLM_ACCUMULATION
            scaler.scale(loss).backward()
            losses.append(float(loss.detach()) * MLM_ACCUMULATION)
            if step % MLM_ACCUMULATION == 0 or step == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                old_scale = scaler.get_scale(); scaler.step(optimizer); scaler.update()
                if scaler.get_scale() >= old_scale:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if step % 50 == 0:
                progress.set_postfix(loss=f"{np.mean(losses[-50:]):.4f}")
        row = {"epoch": epoch, "mlm_loss": float(np.mean(losses))}
        history.append(row)
        print(f"V73 unsupervised epoch={epoch} mlm_loss={row['mlm_loss']:.5f}",
              flush=True)
    # Explicitly save all adapted top-layer tensors (state_dict tensors do not
    # retain the Parameter.requires_grad flag).
    first = len(model.deberta.encoder.layer) - TOP_ADAPTED_LAYERS
    state = {name: value.detach().cpu() for name, value in model.state_dict().items()
             if (name.startswith("deberta.encoder.layer.")
                 and int(name.split(".")[3]) >= first)
             or not name.startswith("deberta.")}
    torch.save({"training_version": TRAINING_VERSION, "model": MODEL_NAME,
                "top_adapted_layers": TOP_ADAPTED_LAYERS,
                "history": history, "state_dict": state}, CHECKPOINT)
    return history


def _probe(c_value=.1):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=float(c_value), class_weight="balanced",
                           max_iter=2500, solver="lbfgs"),
    )


def _evidence(record, risk, decoder):
    spans = decode_model_evidence(
        record["text"], record["offsets"], record["start"], record["end"],
        threshold=float(decoder["threshold"]), max_tokens=int(decoder["max_tokens"]),
        end_policy=str(decoder["end_policy"]), limit=5)
    return apply_evidence_policy(record["text"], int(risk), spans,
                                 policy=str(decoder["cue_policy"]),
                                 topk=int(decoder["topk"]))


def _polarity(records, prediction, evidence):
    prediction = prediction.copy(); evidence = list(evidence)
    for i, record in enumerate(records):
        candidate = polarity_candidate(record["text"], int(prediction[i]))
        if candidate is not None:
            prediction[i], evidence[i] = candidate
    return prediction, evidence


def _metric(truth, prediction, evidence, records):
    risk = float(f1_score(truth, prediction, average="weighted", zero_division=0))
    phrase = float(np.mean([_post_phrase_f1(x, row["gold"])
                            for x, row in zip(evidence, records)]))
    return {"risk_f1": risk, "phrase_f1": phrase,
            "task1": task1_score(risk, phrase)}


def _override(base, probability, threshold):
    proposed = probability.argmax(1); confidence = probability.max(1)
    prediction = base.copy()
    change = (proposed != base) & (confidence >= float(threshold))
    prediction[change] = proposed[change]
    return prediction


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("V73 TAPT requires CUDA")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    seed_everything(config.SEED + 7373)
    frame = load_train_data().reset_index(drop=True)
    labels = frame.risk_label.to_numpy(dtype=np.int64)
    groups = frame.anon_user_id.astype(str).to_numpy()
    outer_train, outer_valid = next(StratifiedGroupKFold(
        n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED,
    ).split(np.zeros(len(frame)), labels, groups))
    nested_records, membership_map, outer_raw = _load_records()
    inner_fit = np.asarray([i for i in outer_train
                            if membership_map[int(i)] != INNER_CAL_FOLD])
    inner_cal = np.asarray([i for i in outer_train
                            if membership_map[int(i)] == INNER_CAL_FOLD])
    nested_by_index = {int(row["global_index"]): row for row in nested_records}
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True,
                                               local_files_only=True)
    dataset = HeadTailDataset(frame.text.astype(str).tolist(), tokenizer)
    device = torch.device("cuda")
    model = OriginalDebertaMLM().to(device)

    control_embeddings = _encode(model, dataset, np.arange(len(frame)), device,
                                 "V73 control neural embeddings")
    history = _train_tapt(model, dataset, outer_train, tokenizer, device)
    adapted_embeddings = _encode(model, dataset, np.arange(len(frame)), device,
                                 "V73 TAPT neural embeddings")
    np.savez_compressed(EMBEDDINGS, control=control_embeddings.astype(np.float16),
                        adapted=adapted_embeddings.astype(np.float16))
    del model; torch.cuda.empty_cache()

    decoder = json.loads((config.OUTPUT_DIR / "task1_evidence_v4" / "calibration.json")
                         .read_text(encoding="utf-8"))
    inner_records = [nested_by_index[int(i)] for i in inner_cal]
    inner_base = np.asarray([int(row["risk"]) for row in inner_records])
    inner_base_evidence = [_evidence(row, risk, decoder)
                           for row, risk in zip(inner_records, inner_base)]
    inner_base, inner_base_evidence = _polarity(
        inner_records, inner_base, inner_base_evidence)
    c_values = (.01, .03, .1, .3, 1.0)
    thresholds = (.50, .60, .70, .80, .90)
    selections = []
    for name, embeddings in (("control", control_embeddings),
                             ("tapt", adapted_embeddings)):
        rows = []
        for c_value in c_values:
            probe = _probe(c_value).fit(embeddings[inner_fit], labels[inner_fit])
            probability = probe.predict_proba(embeddings[inner_cal])
            standalone = probability.argmax(1)
            standalone_f1 = float(f1_score(labels[inner_cal], standalone,
                                            average="weighted", zero_division=0))
            for threshold in thresholds:
                prediction = _override(inner_base, probability, threshold)
                evidence = [_evidence(row, risk, decoder)
                            for row, risk in zip(inner_records, prediction)]
                prediction, evidence = _polarity(inner_records, prediction, evidence)
                score = _metric(labels[inner_cal], prediction, evidence, inner_records)
                rows.append({"representation": name, "c": c_value,
                             "threshold": threshold,
                             "standalone_risk_f1": standalone_f1,
                             **score,
                             "changed": int((prediction != inner_base).sum())})
        selections.append(max(rows, key=lambda row: (row["task1"], row["risk_f1"])))
    selected = max(selections, key=lambda row: (row["task1"], row["risk_f1"]))
    print("V73 inner selections:", json.dumps(selections, indent=2), flush=True)

    embeddings = adapted_embeddings if selected["representation"] == "tapt" else control_embeddings
    probe = _probe(selected["c"]).fit(embeddings[outer_train], labels[outer_train])
    outer_probability = probe.predict_proba(embeddings[outer_valid])
    outer_standalone = outer_probability.argmax(1)
    records = outer_raw["records"]
    base = np.asarray([int(row["risk"]) for row in records])
    base_evidence = [_evidence(row, risk, decoder) for row, risk in zip(records, base)]
    base, base_evidence = _polarity(records, base, base_evidence)
    candidate = _override(base, outer_probability, selected["threshold"])
    candidate_evidence = [_evidence(row, risk, decoder)
                          for row, risk in zip(records, candidate)]
    candidate, candidate_evidence = _polarity(records, candidate, candidate_evidence)
    truth = labels[outer_valid]
    baseline_metric = _metric(truth, base, base_evidence, records)
    candidate_metric = _metric(truth, candidate, candidate_evidence, records)

    # User-cluster bootstrap on the untouched outer split.
    unique = np.unique(groups[outer_valid]); rng = np.random.default_rng(config.SEED + 7373)
    deltas = []
    for _ in range(4000):
        sampled = rng.choice(unique, len(unique), replace=True)
        positions = np.concatenate([np.flatnonzero(groups[outer_valid] == user)
                                    for user in sampled])
        old = _metric(truth[positions], base[positions],
                      [base_evidence[i] for i in positions],
                      [records[i] for i in positions])["task1"]
        new = _metric(truth[positions], candidate[positions],
                      [candidate_evidence[i] for i in positions],
                      [records[i] for i in positions])["task1"]
        deltas.append(new - old)
    deltas = np.asarray(deltas)
    bootstrap = {"mean_delta": float(deltas.mean()),
                 "p05_delta": float(np.quantile(deltas, .05)),
                 "p95_delta": float(np.quantile(deltas, .95)),
                 "positive_fraction": float((deltas > 0).mean())}
    adopted = bool(selected["representation"] == "tapt"
                   and candidate_metric["task1"] >= baseline_metric["task1"] + .002
                   and bootstrap["positive_fraction"] >= .80
                   and bootstrap["p05_delta"] >= 0)
    payload = {
        "training_version": TRAINING_VERSION,
        "method": {"model": MODEL_NAME, "objective": "masked language modelling",
                   "unlabelled_scope": "outer-training Reddit posts only",
                   "max_length": MAX_LENGTH, "epochs": MLM_EPOCHS,
                   "top_adapted_layers": TOP_ADAPTED_LAYERS,
                   "downstream": "frozen CLS+mean+max neural embeddings; balanced LR probe"},
        "evaluation": "inner user calibration; untouched 330-post outer user holdout",
        "mlm_history": history,
        "inner_best_by_representation": selections,
        "selected": selected,
        "outer_standalone_risk_f1": float(f1_score(
            truth, outer_standalone, average="weighted", zero_division=0)),
        "baseline": baseline_metric,
        "candidate": {**candidate_metric,
                      "changed": int((candidate != base).sum()),
                      "confusion": confusion_matrix(truth, candidate,
                                                     labels=np.arange(4)).tolist()},
        "bootstrap": bootstrap,
        "adopted": adopted,
    }
    RESULTS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    return payload


if __name__ == "__main__":
    main()
