"""Gold-rationale crop augmentation gate for Task 1.

Training posts remain user-disjoint from evaluation.  Behavior posts add one
evidence-centred view and Attempt posts add two views, targeting the dominant
Attempt->Behavior error without inventing labels or editing source text.
"""
from __future__ import annotations

import copy
import json

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold

from analyze_task1_lexical_v11 import _lexical_experts, _transformer_probability
from baseline import _post_phrase_f1
from configs.config import config
from datasets.dataset import SuicideRiskDataset
from preprocess.preprocess import load_train_data
from trainer.task1_atomic_v25 import _load_records
from trainer.task1_large_v46 import _risk_probability
from trainer.task1_seed_evidence_v28 import _decode
from trainer.task1_seed_ensemble_v14 import _collect_seed, _criterion
from trainer.train import _loader, _move
from models.multitask_model import SuicideRiskMultiTaskModel, get_optimizer_parameters
from torch.optim import AdamW
from tqdm import tqdm
from utils.seed import seed_everything
from utils.task1_metric import task1_score


OUTPUT = config.OUTPUT_DIR / "task1_rationale_augment_v52"
RESULTS = OUTPUT / "results.json"
CHECKPOINT = OUTPUT / "strict_model.pt"
PREDICTIONS = OUTPUT / "strict_predictions.pt"
FULL_CHECKPOINT = OUTPUT / "full_model.pt"
FULL_MANIFEST = OUTPUT / "full_manifest.json"
TRAINING_VERSION = "task1-gold-rationale-crop-augment-v52"
SEED = 525252


def _crop(record, width, number):
    scores = record["token_labels"].sum(1)
    chunk = int(scores.argmax())
    positive = torch.nonzero(record["token_labels"][chunk] > 0, as_tuple=False).flatten()
    if not len(positive):
        return None
    valid = int(record["attention_mask"][chunk].sum())
    if valid <= 2:
        return None
    content = max(2, width - 2)
    centre = int(torch.median(positive.float()).item())
    start = max(1, centre - content // 2)
    end = min(valid - 1, start + content)
    start = max(1, end - content)
    selected = slice(start, end)
    pad_id = int(record["input_ids"][chunk, valid:].flatten()[0]) if valid < config.MAX_LENGTH else 0
    ids = torch.full((config.MAX_CHUNKS, config.MAX_LENGTH), pad_id, dtype=torch.long)
    mask = torch.zeros_like(ids)
    start_y = torch.zeros((config.MAX_CHUNKS, config.MAX_LENGTH), dtype=torch.float32)
    end_y = torch.zeros_like(start_y); token_y = torch.zeros_like(start_y)
    offsets = [[(0, 0)] * config.MAX_LENGTH for _ in range(config.MAX_CHUNKS)]
    body_ids = record["input_ids"][chunk, selected]
    length = int(body_ids.numel()) + 2
    ids[0, 0] = record["input_ids"][chunk, 0]
    ids[0, 1:length - 1] = body_ids
    ids[0, length - 1] = record["input_ids"][chunk, valid - 1]
    mask[0, :length] = 1
    start_y[0, 1:length - 1] = record["start_labels"][chunk, selected]
    end_y[0, 1:length - 1] = record["end_labels"][chunk, selected]
    token_y[0, 1:length - 1] = record["token_labels"][chunk, selected]
    old_offsets = record["offset_mapping"][chunk][start:end]
    offsets[0][:length] = [(0, 0), *old_offsets, (0, 0)]
    item = copy.copy(record)
    item.update({"row_id": f"{record['row_id']}__rationale_{number}",
                 "input_ids": ids, "attention_mask": mask,
                 "start_labels": start_y, "end_labels": end_y,
                 "token_labels": token_y, "offset_mapping": offsets})
    return item


def _augment(dataset, train_idx):
    original = len(dataset.data); added = []
    for index in map(int, train_idx):
        record = dataset.data[index]; risk = int(record["risk_label"])
        widths = (128,) if risk == 2 else ((96, 160) if risk == 3 else ())
        seen = set()
        for number, width in enumerate(widths):
            item = _crop(record, width, number)
            if item is None:
                continue
            key = tuple(item["input_ids"][0, :int(item["attention_mask"][0].sum())].tolist())
            if key not in seen:
                seen.add(key); added.append(item)
    dataset.data.extend(added)
    indices = np.concatenate((np.asarray(train_idx, dtype=int),
                              np.arange(original, len(dataset.data), dtype=int)))
    return indices, len(added)


def _train(dataset, train_idx, valid_idx, labels):
    if CHECKPOINT.exists() and PREDICTIONS.exists():
        saved = torch.load(PREDICTIONS, map_location="cpu", weights_only=False)
        if np.array_equal(saved["valid_idx"], valid_idx):
            print("V52 resumed", flush=True); return saved["rows"], saved["history"]
    device = torch.device("cuda"); seed_everything(SEED)
    model = SuicideRiskMultiTaskModel().to(device)
    optimizer = AdamW(get_optimizer_parameters(model), weight_decay=config.WEIGHT_DECAY)
    criterion = _criterion(dataset, train_idx, labels, device)
    loader = _loader(dataset, train_idx, True)
    scaler = torch.amp.GradScaler("cuda", enabled=True); history = []
    for epoch in range(1, 4):
        model.train(); optimizer.zero_grad(set_to_none=True); losses = []
        for step, batch in enumerate(tqdm(loader, desc=f"V52 epoch {epoch}/3"), 1):
            batch = _move(batch, device)
            with torch.autocast(device_type="cuda", enabled=True):
                loss = criterion(model(batch["input_ids"], batch["attention_mask"]), batch)["loss"]
                loss /= config.GRADIENT_ACCUMULATION
            scaler.scale(loss).backward(); losses.append(float(loss.detach()) * config.GRADIENT_ACCUMULATION)
            if step % config.GRADIENT_ACCUMULATION == 0 or step == len(loader):
                scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
        history.append({"epoch": epoch, "loss": float(np.mean(losses))})
        print(f"V52 epoch={epoch} loss={history[-1]['loss']:.5f}", flush=True)
    torch.save(model.state_dict(), CHECKPOINT)
    rows = _collect_seed(model, _loader(dataset, valid_idx, False), device)
    torch.save({"valid_idx": valid_idx, "rows": rows, "history": history}, PREDICTIONS)
    return rows, history


def train_full(force=False):
    """Fit the accepted risk expert on all labelled users plus rationale views."""
    if not torch.cuda.is_available():
        raise RuntimeError("V52 full training requires CUDA")
    strict = json.loads(RESULTS.read_text(encoding="utf-8"))
    if not strict.get("risk_only_candidate", {}).get("adopted", False):
        raise RuntimeError("V52 risk-only branch did not pass the strict gate")
    if FULL_CHECKPOINT.exists() and not force:
        print(f"V52 full checkpoint already exists: {FULL_CHECKPOINT}", flush=True)
        return FULL_CHECKPOINT
    OUTPUT.mkdir(parents=True, exist_ok=True)
    dataset = SuicideRiskDataset(config.CACHE_DIR / "train_cache.pt")
    original = len(dataset); all_idx = np.arange(original, dtype=int)
    expanded_idx, added = _augment(dataset, all_idx)
    labels = np.asarray([int(row["risk_label"]) for row in dataset.data])
    device = torch.device("cuda"); seed_everything(SEED)
    model = SuicideRiskMultiTaskModel().to(device)
    optimizer = AdamW(get_optimizer_parameters(model), weight_decay=config.WEIGHT_DECAY)
    criterion = _criterion(dataset, expanded_idx, labels, device)
    loader = _loader(dataset, expanded_idx, True)
    scaler = torch.amp.GradScaler("cuda", enabled=True); history = []
    print(f"V52 full: originals={original}, rationale_views={added}, total={len(expanded_idx)}", flush=True)
    for epoch in range(1, config.FULL_TRAIN_EPOCHS + 1):
        model.train(); optimizer.zero_grad(set_to_none=True); losses = []
        for step, batch in enumerate(tqdm(loader, desc=f"V52 full epoch {epoch}/{config.FULL_TRAIN_EPOCHS}"), 1):
            batch = _move(batch, device)
            with torch.autocast(device_type="cuda", enabled=True):
                loss = criterion(model(batch["input_ids"], batch["attention_mask"]), batch)["loss"]
                loss /= config.GRADIENT_ACCUMULATION
            scaler.scale(loss).backward(); losses.append(float(loss.detach()) * config.GRADIENT_ACCUMULATION)
            if step % config.GRADIENT_ACCUMULATION == 0 or step == len(loader):
                scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
        history.append({"epoch": epoch, "loss": float(np.mean(losses))})
        print(f"V52 full epoch={epoch} loss={history[-1]['loss']:.5f}", flush=True)
    torch.save(model.state_dict(), FULL_CHECKPOINT)
    FULL_MANIFEST.write_text(json.dumps({"training_version": TRAINING_VERSION,
        "original_posts": original, "rationale_views": added,
        "epochs": config.FULL_TRAIN_EPOCHS, "seed": SEED,
        "risk_weight": .10, "history": history}, indent=2), encoding="utf-8")
    print(f"V52 full checkpoint ready: {FULL_CHECKPOINT}", flush=True)
    return FULL_CHECKPOINT


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("V52 requires CUDA")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    dataset = SuicideRiskDataset(config.CACHE_DIR / "train_cache.pt")
    frame = load_train_data().reset_index(drop=True)
    labels = frame.risk_label.to_numpy(dtype=np.int64)
    groups = frame.anon_user_id.astype(str).to_numpy()
    train_idx, valid_idx = next(StratifiedGroupKFold(
        n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED,
    ).split(np.zeros(len(frame)), labels, groups))
    expanded_idx, added = _augment(dataset, train_idx)
    extended_labels = np.asarray([int(row["risk_label"]) for row in dataset.data])
    rows, history = _train(dataset, expanded_idx, valid_idx, extended_labels)

    _, _, raw = _load_records(); records = raw["records"]
    transformer = _transformer_probability(dataset, valid_idx, records)
    v36 = json.loads((config.OUTPUT_DIR / "task1_oof_risk_v36" / "calibration.json").read_text())
    lexical = _lexical_experts(frame, train_idx, valid_idx)[v36["expert"]]
    new_probability = np.vstack([row["probability"] for row in rows])
    texts = frame.text.iloc[valid_idx].astype(str).tolist()
    baseline_risk = _risk_probability(texts, transformer, new_probability, lexical, v36, 0.0)
    candidate_risk = _risk_probability(texts, transformer, new_probability, lexical, v36, 0.10)

    v18 = json.loads((config.OUTPUT_DIR / "task1_candidate_v18" / "results.json").read_text())
    v35 = json.loads((config.OUTPUT_DIR / "task1_oof_decoder_v35" / "calibration.json").read_text())
    params = v35["parameters_by_predicted_risk"] if v35.get("adopted", False) else v18["evidence_parameters_by_predicted_risk"]
    seed2 = torch.load(config.OUTPUT_DIR / "task1_seed_ensemble_v14" / "seed2_valid.pt",
                       map_location="cpu", weights_only=False)["rows"]
    old_s = [record["start"] for record in records]; old_e = [record["end"] for record in records]
    base_s = [.8*a + .2*b["start"] for a,b in zip(old_s,seed2)]
    base_e = [.8*a + .2*b["end"] for a,b in zip(old_e,seed2)]
    new_s = [.7*a + .1*b["start"] + .2*c["start"] for a,b,c in zip(old_s,seed2,rows)]
    new_e = [.7*a + .1*b["end"] + .2*c["end"] for a,b,c in zip(old_e,seed2,rows)]
    old_ev = _decode(records, baseline_risk, base_s, base_e, params)
    risk_only_ev = _decode(records, candidate_risk, base_s, base_e, params)
    new_ev = _decode(records, candidate_risk, new_s, new_e, params)
    gold = [list(frame.iloc[int(i)].evidence) for i in valid_idx]
    old_phrase = np.asarray([_post_phrase_f1(x,y) for x,y in zip(old_ev,gold)])
    risk_only_phrase = np.asarray([_post_phrase_f1(x,y) for x,y in zip(risk_only_ev,gold)])
    new_phrase = np.asarray([_post_phrase_f1(x,y) for x,y in zip(new_ev,gold)])
    truth = labels[valid_idx]
    def metric(risk, phrase):
        rf = float(f1_score(truth,risk,average="weighted",zero_division=0))
        return {"risk_f1":rf,"phrase_f1":float(phrase.mean()),"task1":task1_score(rf,float(phrase.mean()))}
    base = metric(baseline_risk,old_phrase)
    risk_only = metric(candidate_risk,risk_only_phrase)
    candidate = metric(candidate_risk,new_phrase)
    unique=np.unique(groups[valid_idx]); rng=np.random.default_rng(SEED); deltas=[]; risk_only_deltas=[]
    for _ in range(4000):
        users=rng.choice(unique,size=len(unique),replace=True)
        pos=np.concatenate([np.flatnonzero(groups[valid_idx]==u) for u in users])
        a=f1_score(truth[pos],baseline_risk[pos],average="weighted",zero_division=0)
        b=f1_score(truth[pos],candidate_risk[pos],average="weighted",zero_division=0)
        deltas.append(task1_score(b,new_phrase[pos].mean())-task1_score(a,old_phrase[pos].mean()))
        risk_only_deltas.append(task1_score(b,risk_only_phrase[pos].mean())-task1_score(a,old_phrase[pos].mean()))
    deltas=np.asarray(deltas); bootstrap={"mean_delta":float(deltas.mean()),"p05_delta":float(np.quantile(deltas,.05)),"p95_delta":float(np.quantile(deltas,.95)),"positive_fraction":float((deltas>0).mean())}
    risk_only_deltas=np.asarray(risk_only_deltas); risk_only_bootstrap={"mean_delta":float(risk_only_deltas.mean()),"p05_delta":float(np.quantile(risk_only_deltas,.05)),"p95_delta":float(np.quantile(risk_only_deltas,.95)),"positive_fraction":float((risk_only_deltas>0).mean())}
    adopted=bool(candidate["task1"]>=base["task1"]+.003 and bootstrap["positive_fraction"]>=.8)
    risk_only_adopted=bool(risk_only["task1"]>=base["task1"]+.003 and risk_only_bootstrap["positive_fraction"]>=.8)
    payload={"training_version":TRAINING_VERSION,"augmentation":{"added_views":added,"risk_weight":.10,"evidence_weight":.20},"history":history,"baseline":base,"risk_only_candidate":{**risk_only,"changed_risk":int(np.sum(baseline_risk!=candidate_risk)),"bootstrap":risk_only_bootstrap,"adopted":risk_only_adopted},"candidate":{**candidate,"changed_risk":int(np.sum(baseline_risk!=candidate_risk)),"improved_phrase_posts":int((new_phrase>old_phrase).sum()),"worsened_phrase_posts":int((new_phrase<old_phrase).sum()),"confusion":confusion_matrix(truth,candidate_risk,labels=np.arange(4)).tolist()},"user_cluster_bootstrap":bootstrap,"adopted":adopted}
    RESULTS.write_text(json.dumps(payload,indent=2),encoding="utf-8")
    print(json.dumps(payload,indent=2),flush=True); return payload


if __name__ == "__main__":
    main()
