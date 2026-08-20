"""Persona-prompt augmentation for the accepted MentalRoBERTa factor model."""
from __future__ import annotations

import json

import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
from transformers import AutoTokenizer

from configs.config import config
from inference.factor_nli import _rank_decode
from models.factor_model import MentalRobertaFactorModel, factor_optimizer_parameters
from preprocess.preprocess import load_train_data
from trainer.factor_count_aux_v9 import _v3_with_replaced_semantic
from trainer.factor_llm_lexical_v6 import _current_v3_probability
from trainer.factor_train import WeightedGroupedASL
from utils.seed import seed_everything


OUTPUT = config.OUTPUT_DIR / "factor_multiview_prompt_v11"
RESULTS = OUTPUT / "fold0_results.json"
CHECKPOINT = OUTPUT / "fold0_model.pt"
TRAINING_VERSION = "factor-mentalroberta-persona-prompt-v11"
EPOCHS = 3
SEED = 111111
PERSPECTIVES = (
    "As a clinical psychologist, identify emotions, thoughts, mental or physical illness, "
    "self-worth, cognition, trauma, self-harm, suicide history, plans and means. Post: ",
    "As a social worker, identify relationships, isolation, support, family conflict, violence, "
    "school, work, money, stressful events, identity and social difficulties. Post: ",
    "As a resilience counsellor, identify received support, coping actions, hope, confidence, "
    "recovery, responsibilities, goals, purpose and reasons for living. Post: ",
)


class PerspectiveDataset(Dataset):
    def __init__(self, frame):
        tokenizer = AutoTokenizer.from_pretrained(config.FACTOR_MODEL_NAME, use_fast=True)
        self.rows = []
        for row in tqdm(frame.itertuples(index=False), total=len(frame), desc="V11 persona tokenization"):
            view_ids, view_masks = [], []
            for prompt in PERSPECTIVES:
                encoded = tokenizer(
                    prompt + str(row.text), max_length=config.MAX_LENGTH,
                    stride=config.STRIDE, truncation=True,
                    return_overflowing_tokens=True, padding="max_length",
                )
                ids, masks = [], []
                for chunk in range(config.MAX_CHUNKS):
                    if chunk < len(encoded["input_ids"]):
                        ids.append(encoded["input_ids"][chunk]); masks.append(encoded["attention_mask"][chunk])
                    else:
                        ids.append([tokenizer.pad_token_id] * config.MAX_LENGTH)
                        masks.append([0] * config.MAX_LENGTH)
                view_ids.append(ids); view_masks.append(masks)
            self.rows.append({
                "input_ids": torch.tensor(view_ids, dtype=torch.long),
                "attention_mask": torch.tensor(view_masks, dtype=torch.long),
                "targets": torch.tensor(row.factor_vector, dtype=torch.float32),
                "counts": torch.tensor(row.factor_counts, dtype=torch.float32),
            })

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def _loader(dataset, indices, shuffle):
    return DataLoader(
        Subset(dataset, list(map(int, indices))), batch_size=config.BATCH_SIZE,
        shuffle=shuffle, num_workers=0, pin_memory=True,
    )


@torch.no_grad()
def _predict(model, loader, device):
    model.eval(); rows = []
    for batch in tqdm(loader, desc="V11 three-view validation"):
        view_probabilities = []
        for view in range(len(PERSPECTIVES)):
            logits = model(
                batch["input_ids"][:, view].to(device, non_blocking=True),
                batch["attention_mask"][:, view].to(device, non_blocking=True),
            )
            view_probabilities.append(torch.sigmoid(logits.float()))
        rows.append(torch.stack(view_probabilities).mean(0).cpu().numpy())
    return np.vstack(rows)


def train_fold0():
    if not torch.cuda.is_available():
        raise RuntimeError("Factor persona V11 requires CUDA")
    OUTPUT.mkdir(parents=True, exist_ok=True); seed_everything(SEED)
    frame = load_train_data().reset_index(drop=True)
    targets = np.vstack(frame.factor_vector.to_numpy()).astype(np.int8)
    risk = frame.risk_label.to_numpy(np.int64)
    groups = frame.anon_user_id.astype(str).to_numpy()
    train_idx, valid_idx = next(StratifiedGroupKFold(
        n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED,
    ).split(np.zeros(len(frame)), risk, groups))
    dataset = PerspectiveDataset(frame)
    device = torch.device("cuda")
    model = MentalRobertaFactorModel(initialise_labels=False).to(device)
    model.load_state_dict(torch.load(
        config.OUTPUT_DIR / "factor_cv" / "fold0_model.pt", map_location="cpu"
    ))
    positive = torch.tensor(targets[train_idx].sum(0), dtype=torch.float32, device=device)
    weights = torch.sqrt((len(train_idx) - positive) / positive.clamp_min(1.)).clamp(1., 12.)
    loss_fn = WeightedGroupedASL(weights)
    parameters = factor_optimizer_parameters(model)
    parameters[0]["lr"] = config.BACKBONE_LR * .30
    parameters[1]["lr"] = config.HEAD_LR * .40
    optimizer = AdamW(parameters, weight_decay=config.WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=config.FP16)
    train_loader = _loader(dataset, train_idx, True)
    valid_loader = _loader(dataset, valid_idx, False)
    prevalence = targets[train_idx].mean(0)
    current, _ = _current_v3_probability()
    baseline_prediction = _rank_decode(current[valid_idx], prevalence, 1.10)
    baseline = float(f1_score(
        targets[valid_idx], baseline_prediction, average="macro", zero_division=0
    ))
    history = []; best = baseline; best_prediction = baseline_prediction
    for epoch in range(1, EPOCHS + 1):
        model.train(); optimizer.zero_grad(set_to_none=True); losses = []
        for step, batch in enumerate(tqdm(train_loader, desc=f"V11 persona epoch {epoch}"), 1):
            # Rotate views at post level. Across three epochs every example is
            # learned through clinical, social and resilience readings.
            view = (step + epoch) % len(PERSPECTIVES)
            ids = batch["input_ids"][:, view].to(device, non_blocking=True)
            mask = batch["attention_mask"][:, view].to(device, non_blocking=True)
            target = batch["targets"].to(device, non_blocking=True)
            counts = batch["counts"].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", enabled=config.FP16):
                logits, semantic = model(ids, mask, return_semantic=True)
                loss = loss_fn(logits, target, counts)
                loss = loss + config.FACTOR_SEMANTIC_LOSS_WEIGHT * loss_fn(semantic, target)
                loss = loss / config.GRADIENT_ACCUMULATION
            scaler.scale(loss).backward(); losses.append(float(loss.detach()) * config.GRADIENT_ACCUMULATION)
            if step % config.GRADIENT_ACCUMULATION == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
        multiview = _predict(model, valid_loader, device)
        candidate_probability = _v3_with_replaced_semantic(multiview, valid_idx)
        candidate_prediction = _rank_decode(candidate_probability, prevalence, 1.10)
        score = float(f1_score(
            targets[valid_idx], candidate_prediction, average="macro", zero_division=0
        ))
        standalone = float(f1_score(
            targets[valid_idx], _rank_decode(multiview, prevalence, 1.10),
            average="macro", zero_division=0,
        ))
        item = {"epoch": epoch, "train_loss": float(np.mean(losses)),
                "standalone_macro_f1": standalone, "candidate_macro_f1": score}
        history.append(item); print(json.dumps(item), flush=True)
        if score > best:
            best = score; best_prediction = candidate_prediction.copy()
            torch.save(model.state_dict(), CHECKPOINT)
    per_label = [{
        "label": config.ID2FACTOR[label],
        "baseline_f1": float(f1_score(targets[valid_idx, label], baseline_prediction[:, label], zero_division=0)),
        "candidate_f1": float(f1_score(targets[valid_idx, label], best_prediction[:, label], zero_division=0)),
    } for label in range(config.NUM_FACTORS)]
    payload = {
        "training_version": TRAINING_VERSION, "strict_fold": 0,
        "perspectives": list(PERSPECTIVES), "baseline_macro_f1": baseline,
        "best_candidate_macro_f1": best, "delta": best - baseline,
        "history": history, "per_label": per_label,
        "promising_for_full_oof": bool(best >= baseline + .005), "adopted": False,
    }
    RESULTS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    return payload


if __name__ == "__main__":
    train_fold0()
