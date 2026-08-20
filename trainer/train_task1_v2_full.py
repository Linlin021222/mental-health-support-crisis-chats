"""Fit the validated Task 1 V2 risk auxiliary model on all labelled posts."""
import numpy as np
import torch
from torch.optim import AdamW
from tqdm import tqdm

from configs.config import config
from datasets.cache_builder import build_cache
from datasets.dataset import SuicideRiskDataset
from models.multitask_model_v2 import SuicideRiskMultiTaskModelV2, v2_optimizer_parameters
from trainer.train_task1_v2 import Task1V2Loss, _loader, _move
from utils.seed import seed_everything


CHECKPOINT = config.OUTPUT_DIR / "task1_v2_full_model.pt"


def train_task1_v2_full():
    seed_everything(config.SEED)
    build_cache(train=True)
    dataset = SuicideRiskDataset(config.CACHE_DIR / "train_cache.pt")
    indices = np.arange(len(dataset))
    labels = np.asarray([int(x["risk_label"]) for x in dataset.data])
    device = torch.device(config.DEVICE)
    model = SuicideRiskMultiTaskModelV2().to(device)
    counts = np.bincount(labels, minlength=4)
    class_weights = np.sqrt(len(dataset) / np.maximum(counts, 1))
    class_weights = torch.tensor(
        class_weights / class_weights.mean(), dtype=torch.float, device=device
    )
    ordinal_targets = labels[:, None] > np.arange(3)[None, :]
    ordinal_positive = ordinal_targets.sum(0)
    ordinal_weight = torch.tensor(
        np.sqrt((len(dataset)-ordinal_positive)/np.maximum(ordinal_positive, 1)),
        dtype=torch.float, device=device,
    )
    factor_targets = torch.stack([x["factor_vector"] for x in dataset.data]).float()
    factor_positive = factor_targets.sum(0)
    factor_weight = torch.sqrt((len(dataset)-factor_positive)/factor_positive.clamp_min(1.0))
    factor_weight = factor_weight.clamp(1.0, 10.0).to(device)
    token_positive = sum(float(x["token_labels"].sum()) for x in dataset.data)
    token_total = sum(float(x["attention_mask"].sum()) for x in dataset.data)
    token_weight = torch.tensor(
        min(15.0, max(3.0, np.sqrt((token_total-token_positive)/max(token_positive, 1.0)))),
        device=device,
    )
    criterion = Task1V2Loss(
        class_weights, ordinal_weight, token_weight, factor_weight
    ).to(device)
    optimizer = AdamW(v2_optimizer_parameters(model), weight_decay=config.WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=config.FP16)
    loader = _loader(dataset, indices, True)
    for epoch in range(1, config.EPOCHS + 1):
        model.train(); optimizer.zero_grad(set_to_none=True); losses = []
        for step, batch in enumerate(tqdm(loader, desc=f"task1-v2-full epoch {epoch}"), 1):
            batch = _move(batch, device)
            with torch.autocast(device_type=device.type, enabled=config.FP16):
                output = model(batch["input_ids"], batch["attention_mask"])
                loss, _ = criterion(output, batch)
                loss = loss / config.GRADIENT_ACCUMULATION
            scaler.scale(loss).backward()
            losses.append(float(loss.detach()) * config.GRADIENT_ACCUMULATION)
            if step % config.GRADIENT_ACCUMULATION == 0 or step == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
        torch.save(model.state_dict(), CHECKPOINT)
        print(f"task1-v2-full epoch={epoch} loss={np.mean(losses):.4f}")
    print(f"Task 1 V2 full checkpoint: {CHECKPOINT}")
    return CHECKPOINT


if __name__ == "__main__":
    train_task1_v2_full()
