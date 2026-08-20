"""Train the full-data artifacts required by the adopted Task 1 V18 ensemble."""
from __future__ import annotations

import json

import joblib
import numpy as np
import torch
from sklearn.svm import LinearSVC
from torch.optim import AdamW
from tqdm import tqdm

from baseline import _vectorizer
from configs.config import config
from datasets.cache_builder import build_cache
from datasets.dataset import SuicideRiskDataset
from models.multitask_model import SuicideRiskMultiTaskModel, get_optimizer_parameters
from preprocess.preprocess import load_train_data
from trainer.task1_seed_ensemble_v14 import SEED2, _criterion
from trainer.train import _loader, _move
from utils.seed import seed_everything


OUTPUT = config.OUTPUT_DIR / "task1_candidate_v18"
CALIBRATION = OUTPUT / "calibration.json"
CHECKPOINT = OUTPUT / "full_seed2_model.pt"
LEXICAL_MODEL = OUTPUT / "full_lexical_svc.joblib"
MANIFEST = OUTPUT / "full_training_manifest.json"


def _fit_lexical_model():
    frame = load_train_data().reset_index(drop=True)
    vectorizer = _vectorizer()
    matrix = vectorizer.fit_transform(frame.text.astype(str))
    model = LinearSVC(C=0.25, class_weight="balanced")
    model.fit(matrix, np.asarray(frame.risk_label, dtype=np.int64))
    joblib.dump({
        "training_version": "task1-consolidated-candidate-v18",
        "vectorizer": vectorizer,
        "risk_model": model,
        "train_posts": int(len(frame)),
    }, LEXICAL_MODEL)
    print(f"V18 lexical expert fitted on all {len(frame)} posts: {LEXICAL_MODEL}", flush=True)


def train_task1_v18_full(force=False):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if not CALIBRATION.exists():
        raise FileNotFoundError("Run --mode task1-candidate-v18 before full V18 training")
    calibration = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    if not calibration.get("adopted", False):
        raise RuntimeError("V18 did not pass its development adoption gate")

    if force or not LEXICAL_MODEL.exists():
        _fit_lexical_model()
    else:
        print(f"V18 lexical expert already exists: {LEXICAL_MODEL}", flush=True)

    cache_file = config.CACHE_DIR / "train_cache.pt"
    if not cache_file.exists():
        build_cache(train=True)
    dataset = SuicideRiskDataset(cache_file)
    expected = (config.MAX_CHUNKS, config.MAX_LENGTH)
    if tuple(dataset.data[0]["input_ids"].shape) != expected:
        build_cache(train=True)
        dataset = SuicideRiskDataset(cache_file)

    if CHECKPOINT.exists() and not force:
        print(f"V18 full second-seed checkpoint already exists: {CHECKPOINT}", flush=True)
        return CHECKPOINT
    if not torch.cuda.is_available():
        raise RuntimeError("Full V18 second-seed training requires CUDA")

    indices = np.arange(len(dataset))
    labels = np.asarray([int(row["risk_label"]) for row in dataset.data])
    device = torch.device("cuda")
    seed_everything(SEED2)
    model = SuicideRiskMultiTaskModel().to(device)
    optimizer = AdamW(get_optimizer_parameters(model), weight_decay=config.WEIGHT_DECAY)
    criterion = _criterion(dataset, indices, labels, device)
    loader = _loader(dataset, indices, True)
    scaler = torch.amp.GradScaler("cuda", enabled=config.FP16)
    history = []
    epochs = int(config.FULL_TRAIN_EPOCHS)
    print(
        f"V18 full second seed: {len(dataset)} posts, seed={SEED2}, epochs={epochs}",
        flush=True,
    )
    for epoch in range(1, epochs + 1):
        model.train(); optimizer.zero_grad(set_to_none=True); losses = []
        progress = tqdm(loader, desc=f"task1-v18-full epoch {epoch}/{epochs}")
        for step, batch in enumerate(progress, 1):
            batch = _move(batch, device)
            with torch.autocast(device_type="cuda", enabled=config.FP16):
                loss = criterion(
                    model(batch["input_ids"], batch["attention_mask"]), batch
                )["loss"] / config.GRADIENT_ACCUMULATION
            scaler.scale(loss).backward()
            losses.append(float(loss.detach()) * config.GRADIENT_ACCUMULATION)
            if step % config.GRADIENT_ACCUMULATION == 0 or step == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
            if step % 100 == 0:
                progress.set_postfix(loss=f"{np.mean(losses[-100:]):.3f}")
        epoch_loss = float(np.mean(losses))
        history.append({"epoch": epoch, "train_loss": epoch_loss})
        print(f"task1-v18-full epoch={epoch} train_loss={epoch_loss:.4f}", flush=True)

    torch.save(model.state_dict(), CHECKPOINT)
    MANIFEST.write_text(json.dumps({
        "training_version": "task1-consolidated-candidate-v18",
        "seed": SEED2,
        "train_posts": len(dataset),
        "epochs": epochs,
        "history": history,
        "checkpoint": str(CHECKPOINT),
        "lexical_model": str(LEXICAL_MODEL),
    }, indent=2), encoding="utf-8")
    print(f"V18 full-data artifacts ready: {CHECKPOINT}", flush=True)
    return CHECKPOINT


if __name__ == "__main__":
    train_task1_v18_full()
