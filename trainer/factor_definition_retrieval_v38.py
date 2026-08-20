"""Strict fold-0 integrated definition/LSFA/HTTN/retrieval experiment."""
from __future__ import annotations

import json

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from configs.config import config
from datasets.factor_cache_builder import build_factor_cache
from models.factor_definition_retrieval_v38 import (
    DefinitionRetrievalFactorModel, encode_paper_definitions,
)
from models.factor_model import MentalRobertaFactorModel
from preprocess.preprocess import load_train_data
from trainer.factor_mhlat_v4 import _current_components
from trainer.factor_train import FactorDataset, WeightedGroupedASL, _collate
from utils.factor_calibration import apply_prior_topk
from utils.seed import seed_everything


OUTPUT = config.OUTPUT_DIR / "factor_definition_retrieval_v38"
RESULTS = OUTPUT / "fold0_results.json"
CHECKPOINT = OUTPUT / "fold0_model.pt"
PREDICTIONS = OUTPUT / "fold0_valid.npz"
FEATURES = OUTPUT / "fold0_base_features.npz"
BASE_CHECKPOINT = config.OUTPUT_DIR / "factor_cv" / "fold0_model.pt"
TOPK = 4
EPOCHS = 2
TAIL_THRESHOLD = 40


class IndexedDataset(Dataset):
    def __init__(self, base, indices, retrieval):
        self.base = base; self.indices = np.asarray(indices); self.retrieval = retrieval
    def __len__(self): return len(self.indices)
    def __getitem__(self, position):
        index = int(self.indices[position])
        return index, self.base[index], self.retrieval[index]


def _collate_indexed(rows):
    batch = _collate([row[1] for row in rows])
    batch["indices"] = torch.tensor([row[0] for row in rows], dtype=torch.long)
    batch["retrieval"] = torch.stack([row[2] for row in rows])
    return batch


def _loader(dataset, indices, retrieval, shuffle):
    return DataLoader(
        IndexedDataset(dataset, indices, retrieval), batch_size=config.BATCH_SIZE,
        shuffle=shuffle, collate_fn=_collate_indexed,
        num_workers=config.NUM_WORKERS, pin_memory=config.DEVICE == "cuda",
    )


@torch.no_grad()
def _base_features(dataset, frame, device):
    if FEATURES.exists():
        saved = np.load(FEATURES)
        if saved["features"].shape[0] == len(dataset):
            return saved["features"].astype(np.float32)
    model = MentalRobertaFactorModel(initialise_labels=False).to(device)
    model.load_state_dict(torch.load(BASE_CHECKPOINT, map_location=device)); model.eval()
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=_collate)
    result = []
    for batch in tqdm(loader, desc="V38 task-specific retrieval features"):
        _, feature = model(
            batch["input_ids"].to(device), batch["attention_mask"].to(device),
            return_features=True,
        )
        result.append(feature.cpu().numpy())
    result = np.vstack(result).astype(np.float32)
    np.savez_compressed(FEATURES, features=result)
    del model
    if device.type == "cuda": torch.cuda.empty_cache()
    return result


def _retrieval_bank(features, groups, train_idx):
    normal = features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-8)
    retrieval = np.zeros((len(features), TOPK, features.shape[1]), dtype=np.float32)
    train_idx = np.asarray(train_idx)
    # One BLAS operation is much faster than one matrix multiplication per
    # document. The 1,635 x 1,635 float32 matrix is only about 11 MB.
    similarity_matrix = normal @ normal.T
    train_mask = np.zeros(len(features), dtype=bool); train_mask[train_idx] = True
    for index in range(len(features)):
        candidate_mask = train_mask & (groups != groups[index])
        candidates = np.flatnonzero(candidate_mask)
        if not len(candidates): candidates = train_idx[train_idx != index]
        similarity = similarity_matrix[index, candidates]
        count = min(TOPK, len(candidates))
        chosen = candidates[np.argpartition(similarity, -count)[-count:]]
        chosen = chosen[np.argsort(similarity_matrix[index, chosen])[::-1]]
        values = features[chosen]
        if len(values) < TOPK:
            values = np.concatenate((values, np.repeat(values[-1:], TOPK-len(values), 0)))
        retrieval[index] = values
    del similarity_matrix
    return torch.tensor(retrieval, dtype=torch.float32)


def _initialise(device, definitions, tail_mask, feature_std):
    model = DefinitionRetrievalFactorModel(definitions, tail_mask, feature_std)
    incompatible = model.load_state_dict(
        torch.load(BASE_CHECKPOINT, map_location="cpu"), strict=False,
    )
    if incompatible.unexpected_keys:
        raise RuntimeError(f"Unexpected base keys: {incompatible.unexpected_keys}")
    allowed_prefixes = (
        "definition_embeddings", "tail_mask", "feature_std", "definition_projection.",
        "definition_gate", "retrieval_projection.", "retrieval_gate", "fusion_norm.",
        "tail_generator.", "tail_gate",
    )
    bad = [key for key in incompatible.missing_keys
           if not key.startswith(allowed_prefixes)]
    if bad: raise RuntimeError(f"Unexpected missing keys: {bad}")
    return model.to(device)


def _auxiliary(aux, targets, tail_mask, head_mask, model):
    rows, labels = torch.where(targets > .5)
    if len(rows):
        definition = F.cross_entropy(aux["definition_logits"][rows, labels], labels)
    else:
        definition = targets.sum() * 0.
    transfer = F.mse_loss(
        aux["generated_weights"][head_mask],
        model.label_weights.detach()[head_mask],
    )
    tail_positive = (targets > .5) & tail_mask.unsqueeze(0)
    if tail_positive.any():
        lsfa = F.softplus(-aux["augmented_logits"][tail_positive]).mean()
    else:
        lsfa = targets.sum() * 0.
    # Confusion boundary: positive labels outrank their most similar absent
    # definition, while genuine co-occurring positives are never penalised.
    definition_normal = F.normalize(model.definition_embeddings, dim=-1)
    similarity = definition_normal @ definition_normal.T
    similarity.fill_diagonal_(-1e4)
    hard = similarity.argmax(-1)
    margins = []
    logits = aux["augmented_logits"]
    for label in range(config.NUM_FACTORS):
        negative = int(hard[label]); active = (targets[:, label] > .5) & (targets[:, negative] < .5)
        if active.any():
            margins.append(F.relu(.2 - logits[active, label] + logits[active, negative]).mean())
    margin = torch.stack(margins).mean() if margins else targets.sum() * 0.
    return definition, transfer, lsfa, margin


def _train(model, loader, loss_fn, optimizer, scaler, device, tail_mask, head_mask, epoch):
    model.train(); optimizer.zero_grad(set_to_none=True); losses = []
    for step, batch in enumerate(tqdm(loader, desc=f"V38 integrated epoch {epoch}"), 1):
        targets = batch["factor_vectors"].to(device); counts = batch["factor_counts"].to(device)
        with torch.autocast(device_type=device.type, enabled=config.FP16):
            logits, aux = model(
                batch["input_ids"].to(device), batch["attention_mask"].to(device),
                batch["retrieval"].to(device), return_aux=True,
            )
            classification = loss_fn(logits, targets, counts)
            semantic = loss_fn(aux["semantic_logits"], targets)
            definition, transfer, lsfa, margin = _auxiliary(
                aux, targets, tail_mask, head_mask, model,
            )
            loss = (classification + .08*semantic + .05*definition + .02*transfer
                    + .03*lsfa + .04*margin) / config.GRADIENT_ACCUMULATION
        scaler.scale(loss).backward(); losses.append(float(loss.detach())*config.GRADIENT_ACCUMULATION)
        if step % config.GRADIENT_ACCUMULATION == 0 or step == len(loader):
            scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
    return float(np.mean(losses))


@torch.no_grad()
def _probabilities(model, loader, device):
    model.eval(); values = []
    for batch in loader:
        logits = model(
            batch["input_ids"].to(device), batch["attention_mask"].to(device),
            batch["retrieval"].to(device),
        )
        values.append(torch.sigmoid(logits).cpu().numpy())
    return np.vstack(values)


def main():
    if not torch.cuda.is_available(): raise RuntimeError("V38 requires CUDA")
    OUTPUT.mkdir(parents=True, exist_ok=True); seed_everything(config.SEED + 3838)
    cache = build_factor_cache(train=True); dataset = FactorDataset(cache)
    frame = load_train_data().reset_index(drop=True)
    current, targets, calibration = _current_components()
    groups = frame.anon_user_id.astype(str).to_numpy()
    folds = list(StratifiedGroupKFold(
        n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED,
    ).split(np.zeros(len(frame)), frame.risk_label, groups))
    train_idx, valid_idx = folds[0]; device = torch.device(config.DEVICE)
    features = _base_features(dataset, frame, device)
    retrieval = _retrieval_bank(features, groups, train_idx)

    positive = targets[train_idx].sum(0)
    tail_mask = torch.tensor(positive < TAIL_THRESHOLD, dtype=torch.bool, device=device)
    head_mask = ~tail_mask
    base = MentalRobertaFactorModel(initialise_labels=False).to(device)
    base.load_state_dict(torch.load(BASE_CHECKPOINT, map_location=device))
    definitions = encode_paper_definitions(base, device)
    del base; torch.cuda.empty_cache()
    feature_std = torch.tensor(features[train_idx].std(0), dtype=torch.float32, device=device)
    model = _initialise(device, definitions, tail_mask, feature_std)

    weights = torch.sqrt((len(train_idx)-torch.tensor(positive, device=device))
                         / torch.tensor(positive, device=device).clamp_min(1.)).clamp(1.,12.)
    loss_fn = WeightedGroupedASL(weights).to(device)
    backbone = [p for n,p in model.named_parameters() if n.startswith("encoder.")]
    head = [p for n,p in model.named_parameters() if not n.startswith("encoder.")]
    optimizer = AdamW([{"params":backbone,"lr":3e-6},{"params":head,"lr":2e-5}],
                      weight_decay=config.WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=config.FP16)
    train_loader = _loader(dataset, train_idx, retrieval, True)
    valid_loader = _loader(dataset, valid_idx, retrieval, False)
    prevalence = targets[train_idx].mean(0)
    initial = _probabilities(model, valid_loader, device)
    best_probability = initial; best = {"epoch":0,"score":float(f1_score(
        targets[valid_idx], apply_prior_topk(initial,prevalence,1.10),
        average="macro",zero_division=0))}
    for epoch in range(1,EPOCHS+1):
        loss = _train(model,train_loader,loss_fn,optimizer,scaler,device,tail_mask,head_mask,epoch)
        probability = _probabilities(model,valid_loader,device)
        score = float(f1_score(targets[valid_idx],apply_prior_topk(probability,prevalence,1.10),
                               average="macro",zero_division=0))
        print(f"V38 epoch={epoch} loss={loss:.5f} standalone_macro_f1={score:.5f}",flush=True)
        if score>best["score"]:
            best={"epoch":epoch,"score":score,"loss":loss}; best_probability=probability.copy()
            torch.save(model.state_dict(),CHECKPOINT)

    baseline_prediction=apply_prior_topk(current[valid_idx],prevalence,float(calibration["prevalence_ratio"]))
    baseline=float(f1_score(targets[valid_idx],baseline_prediction,average="macro",zero_division=0))
    grid=[]
    for weight in (0.,.05,.10,.15,.20,.30):
        mixed=(1-weight)*current[valid_idx]+weight*best_probability
        for ratio in (1.0,1.10):
            prediction=apply_prior_topk(mixed,prevalence,ratio)
            grid.append({"weight":weight,"ratio":ratio,"macro_f1":float(f1_score(
                targets[valid_idx],prediction,average="macro",zero_division=0))})
    grid.sort(key=lambda x:x["macro_f1"],reverse=True)
    # Fixed 10% residual at unchanged decoder ratio is the non-optimistic test.
    fixed_probability=.9*current[valid_idx]+.1*best_probability
    fixed_prediction=apply_prior_topk(fixed_probability,prevalence,float(calibration["prevalence_ratio"]))
    fixed=float(f1_score(targets[valid_idx],fixed_prediction,average="macro",zero_division=0))
    per_label=[]
    for label in range(config.NUM_FACTORS):
        old=f1_score(targets[valid_idx,label],baseline_prediction[:,label],zero_division=0)
        new=f1_score(targets[valid_idx,label],fixed_prediction[:,label],zero_division=0)
        per_label.append({"label":config.ID2FACTOR[label],"support":int(targets[valid_idx,label].sum()),
                          "tail":bool(tail_mask[label]),"baseline_f1":float(old),
                          "candidate_f1":float(new),"delta":float(new-old)})
    payload={"training_version":"factor-integrated-definition-retrieval-v38",
             "evaluation_scope":"strict user-disjoint fold0",
             "components":{"formal_definition_token_alignment":True,
                           "retrieval_cross_attention_topk":TOPK,
                           "lsfa_tail_positive_pair_augmentation":True,
                           "httn_head_to_tail_classifier_transfer":True,
                           "confusion_boundary_margin":True},
             "tail_threshold":TAIL_THRESHOLD,"tail_labels":int(tail_mask.sum()),
             "transferred_initial":best if best["epoch"]==0 else {"epoch":0,"score":float(f1_score(
                 targets[valid_idx],apply_prior_topk(initial,prevalence,1.10),average="macro",zero_division=0))},
             "best_standalone":best,"current_baseline_macro_f1":baseline,
             "fixed_10pct_macro_f1":fixed,"fixed_delta":fixed-baseline,
             "optimistic_grid_top10":grid[:10],"per_label_fixed":per_label,
             "promising":bool(fixed>=baseline+.003)}
    np.savez_compressed(PREDICTIONS,valid_indices=valid_idx,probabilities=best_probability)
    RESULTS.write_text(json.dumps(payload,indent=2),encoding="utf-8")
    print(json.dumps(payload,indent=2),flush=True); return payload


if __name__=="__main__": main()
