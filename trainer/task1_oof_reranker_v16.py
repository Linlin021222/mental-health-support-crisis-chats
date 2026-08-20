"""OOF hard-negative evidence reranker with two levels of user isolation."""
from __future__ import annotations

import json

import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from configs.config import config
from datasets.dataset import SuicideRiskDataset
from inference.task1_evidence_v4 import correct_risk_only, load_evidence_calibration
from models.multitask_model import SuicideRiskMultiTaskModel, get_optimizer_parameters
from trainer.task1_evidence_reranker_v13 import (
    PairDataset, _dedupe_ranked, _evaluate_grid, _make_model, _pair_rows,
    _pool_records, _post_score_map, _score, _strict_baseline,
)
from trainer.task1_seed_ensemble_v14 import _criterion
from trainer.train import _loader, _move
from utils.seed import seed_everything
from utils.task1_metric import task1_score
from baseline import _post_phrase_f1


OUTPUT = config.OUTPUT_DIR / "task1_oof_reranker_v16"
INNER_MODEL = OUTPUT / "inner_first_stage.pt"
INNER_RAW = OUTPUT / "inner_oof_raw.pt"
RERANKER = OUTPUT / "reranker.pt"
RESULTS = OUTPUT / "results.json"
CALIBRATION = OUTPUT / "calibration.json"
TRAINING_VERSION = "task1-oof-context-reranker-v16"


@torch.no_grad()
def _collect_first_stage(model, dataset, indices, device):
    rows = []; loader = _loader(dataset, indices, False); cursor = 0
    model.eval()
    for batch in tqdm(loader, desc="v16 inner OOF candidates"):
        output = model(
            batch["input_ids"].to(device), batch["attention_mask"].to(device)
        )
        probability = torch.softmax(output["risk_logits"], -1).cpu().numpy()
        for i in range(len(batch["row_id"])):
            source = dataset.data[int(indices[cursor])]
            risk = correct_risk_only(batch["texts"][i], int(np.argmax(probability[i])))
            rows.append({
                "row_id": str(batch["row_id"][i]), "user": str(source["anon_user_id"]),
                "text": batch["texts"][i], "offsets": batch["offset_mappings"][i],
                "truth": int(batch["risk_labels"][i]), "gold": batch["evidences"][i],
                "risk": risk, "old_probability": probability[i],
                "start": output["start_logits"][i].cpu(),
                "end": output["end_logits"][i].cpu(),
            })
            cursor += 1
    return rows


def _inner_oof_records(dataset, outer_train, labels, groups, device):
    if INNER_RAW.exists():
        saved = torch.load(INNER_RAW, map_location="cpu", weights_only=False)
        print("v16: resumed genuine OOF first-stage candidates", flush=True)
        return saved["records"]
    local_labels = labels[outer_train]; local_groups = groups[outer_train]
    inner_fit_local, inner_oof_local = next(StratifiedGroupKFold(
        n_splits=4, shuffle=True, random_state=config.SEED + 1616
    ).split(np.zeros(len(outer_train)), local_labels, local_groups))
    inner_fit = outer_train[inner_fit_local]; inner_oof = outer_train[inner_oof_local]
    seed_everything(config.SEED + 1616)
    model = SuicideRiskMultiTaskModel().to(device)
    optimizer = AdamW(get_optimizer_parameters(model), weight_decay=config.WEIGHT_DECAY)
    criterion = _criterion(dataset, inner_fit, labels, device)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    loader = _loader(dataset, inner_fit, True)
    print(
        f"v16: inner first stage trains on {len(inner_fit)} posts; "
        f"OOF candidates from {len(inner_oof)} unseen posts", flush=True,
    )
    history = []
    for epoch in range(1, 4):
        model.train(); optimizer.zero_grad(set_to_none=True); losses = []
        for step, batch in enumerate(tqdm(loader, desc=f"v16 first-stage epoch {epoch}/3"), 1):
            batch = _move(batch, device)
            with torch.autocast(device_type="cuda", enabled=True):
                loss = criterion(
                    model(batch["input_ids"], batch["attention_mask"]), batch
                )["loss"] / config.GRADIENT_ACCUMULATION
            scaler.scale(loss).backward(); losses.append(float(loss.detach()) * config.GRADIENT_ACCUMULATION)
            if step % config.GRADIENT_ACCUMULATION == 0 or step == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
        history.append(float(np.mean(losses)))
        print(f"v16 first-stage epoch={epoch} loss={history[-1]:.4f}", flush=True)
    torch.save(model.state_dict(), INNER_MODEL)
    records = _collect_first_stage(model, dataset, inner_oof, device)
    torch.save({
        "inner_fit": inner_fit, "inner_oof": inner_oof,
        "history": history, "records": records,
    }, INNER_RAW)
    del model, optimizer; torch.cuda.empty_cache()
    return records


def _train_reranker(oof_posts, tokenizer, device):
    users = np.asarray([post["user"] for post in oof_posts])
    fit_posts, calibration_posts = next(GroupShuffleSplit(
        n_splits=1, test_size=0.25, random_state=config.SEED + 1617
    ).split(np.arange(len(oof_posts)), groups=users))
    fit_rows = _pair_rows(oof_posts, fit_posts, training=True)
    calibration_rows = _pair_rows(oof_posts, calibration_posts, training=False)
    fit_dataset = PairDataset(fit_rows, tokenizer)
    calibration_dataset = PairDataset(calibration_rows, tokenizer)
    model = _make_model().to(device)
    positive = sum(row["label"] for row in fit_rows); negative = len(fit_rows) - positive
    pos_weight = torch.tensor(
        min(6.0, max(1.0, negative / max(positive, 1.0))), device=device
    )
    backbone = [p for n, p in model.named_parameters() if n.startswith("deberta.") and p.requires_grad]
    head = [p for n, p in model.named_parameters() if not n.startswith("deberta.") and p.requires_grad]
    optimizer = AdamW([
        {"params": backbone, "lr": 3e-6}, {"params": head, "lr": 2e-5},
    ], weight_decay=config.WEIGHT_DECAY)
    loader = DataLoader(fit_dataset, batch_size=4, shuffle=True, num_workers=0)
    scaler = torch.amp.GradScaler("cuda", enabled=True); best = None; history = []
    for epoch in range(1, 5):
        model.train(); optimizer.zero_grad(set_to_none=True); losses = []
        for step, batch in enumerate(tqdm(loader, desc=f"v16 reranker epoch {epoch}/4"), 1):
            with torch.autocast(device_type="cuda", enabled=True):
                logits = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                ).logits.squeeze(-1)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, batch["label"].to(device), pos_weight=pos_weight
                ) / 4.0
            scaler.scale(loss).backward(); losses.append(float(loss.detach()) * 4.0)
            if step % 4 == 0 or step == len(loader):
                scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
        scores = _score(model, calibration_dataset, device, "v16 OOF calibration")
        selected = _evaluate_grid(
            oof_posts, calibration_rows, scores, calibration_posts
        )[0]
        row = {
            "epoch": epoch, "loss": float(np.mean(losses)),
            **{key: selected[key] for key in ("topk", "threshold", "phrase_f1")},
        }
        history.append(row)
        print(
            f"v16 reranker epoch={epoch} OOF_phrase={row['phrase_f1']:.4f} "
            f"topk={row['topk']} threshold={row['threshold']:.2f}", flush=True,
        )
        if best is None or row["phrase_f1"] > best["phrase_f1"]:
            best = row; torch.save(model.state_dict(), RERANKER)
    model.load_state_dict(torch.load(RERANKER, map_location=device))
    return model, best, history, len(fit_rows)


def train_task1_oof_reranker_v16():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("V16 requires CUDA")
    dataset = SuicideRiskDataset(config.CACHE_DIR / "train_cache.pt")
    labels = np.asarray([int(row["risk_label"]) for row in dataset.data])
    groups = np.asarray([str(row["anon_user_id"]) for row in dataset.data])
    outer_train, outer_valid = next(StratifiedGroupKFold(
        n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED
    ).split(np.zeros(len(dataset)), labels, groups))
    strict_raw = torch.load(
        config.OUTPUT_DIR / "task1_evidence_v4" / "fold0_raw.pt",
        map_location="cpu", weights_only=False,
    )
    if not np.array_equal(strict_raw["valid_idx"], outer_valid):
        raise ValueError("V16 and V4 outer folds differ")
    device = torch.device("cuda")
    inner_records = _inner_oof_records(
        dataset, outer_train, labels, groups, device
    )
    oof_posts = _pool_records(inner_records, use_truth=False)
    strict_posts = _pool_records(strict_raw["records"], use_truth=False)
    tokenizer = AutoTokenizer.from_pretrained(
        config.MODEL_NAME, use_fast=True, local_files_only=True
    )
    model, selected, history, fit_pairs = _train_reranker(oof_posts, tokenizer, device)
    strict_rows = _pair_rows(strict_posts, np.arange(len(strict_posts)), training=False)
    strict_dataset = PairDataset(strict_rows, tokenizer)
    strict_pair_scores = _score(model, strict_dataset, device, "v16 outer strict")
    grouped = _post_score_map(strict_rows, strict_pair_scores)
    scores = []; predictions = []
    for index, post in enumerate(strict_posts):
        values = [grouped[index].get(i, 0.0) for i in range(len(post["candidates"]))]
        evidence = [] if post["risk"] == 0 else _dedupe_ranked(
            post, values, selected["topk"], selected["threshold"]
        )
        predictions.append(evidence); scores.append(_post_phrase_f1(evidence, post["gold"]))
    scores = np.asarray(scores, dtype=np.float32)
    evidence_calibration = load_evidence_calibration()
    baseline = _strict_baseline(strict_raw["records"], evidence_calibration)
    risk_f1 = float(evidence_calibration["strict_risk_f1"])
    local_groups = groups[outer_valid]; unique = np.unique(local_groups)
    rng = np.random.default_rng(config.SEED + 1618); deltas = []
    for _ in range(3000):
        sampled_users = rng.choice(unique, size=len(unique), replace=True)
        sampled = np.concatenate([
            np.flatnonzero(local_groups == user) for user in sampled_users
        ])
        deltas.append(float(scores[sampled].mean() - baseline[sampled].mean()))
    bootstrap = {
        "mean_phrase_delta": float(np.mean(deltas)),
        "p05_phrase_delta": float(np.quantile(deltas, 0.05)),
        "p95_phrase_delta": float(np.quantile(deltas, 0.95)),
        "positive_fraction": float(np.mean(np.asarray(deltas) > 0)),
    }
    phrase_f1 = float(scores.mean()); baseline_phrase = float(baseline.mean())
    adopted = bool(
        phrase_f1 >= baseline_phrase + 0.005
        and bootstrap["positive_fraction"] >= 0.80
    )
    payload = {
        "training_version": TRAINING_VERSION,
        "inner_oof_posts": len(oof_posts), "reranker_fit_pairs": fit_pairs,
        "reranker_history": history, "selected": selected,
        "baseline": {
            "risk_f1": risk_f1, "phrase_f1": baseline_phrase,
            "task1": task1_score(risk_f1, baseline_phrase),
        },
        "strict": {
            "risk_f1": risk_f1, "phrase_f1": phrase_f1,
            "task1": task1_score(risk_f1, phrase_f1),
            "improved_posts": int((scores > baseline).sum()),
            "worsened_posts": int((scores < baseline).sum()),
        },
        "user_cluster_bootstrap": bootstrap, "adopted": adopted,
    }
    RESULTS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    CALIBRATION.write_text(json.dumps({
        "training_version": TRAINING_VERSION, "adopted": adopted,
        "topk": selected["topk"], "threshold": selected["threshold"],
        "strict_phrase_f1": phrase_f1,
        "strict_task1": task1_score(risk_f1, phrase_f1),
        "bootstrap_positive_fraction": bootstrap["positive_fraction"],
    }, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    return payload


if __name__ == "__main__":
    train_task1_oof_reranker_v16()
