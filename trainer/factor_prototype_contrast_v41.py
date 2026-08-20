"""Train the accepted prototype cross-encoder on V40 grounded hypotheses.

Each positive factor is paired with an absent confusable label. Both receive a
cross-user retrieved grounding example, and a pairwise margin teaches the NLI
backbone the taxonomy boundary instead of relying on inference-only prompting.
"""
from __future__ import annotations

import json
import random

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_cosine_schedule_with_warmup

from configs.config import config
from inference.factor_nli import _entailment_index, _rank_decode
from preprocess.factor_prototypes import FACTOR_PROTOTYPES
from preprocess.preprocess import load_train_data
from trainer.factor_cross_encoder_v2 import CONFUSION_GROUPS
from trainer.factor_mhlat_v4 import _current_components
from trainer.factor_prototype_retrieval_v40 import (
    _best_excerpt, _guided_hypothesis, _predict, _prototype_weights, _vectorizer,
)
from utils.seed import seed_everything


OUTPUT = config.OUTPUT_DIR / "factor_prototype_contrast_v41"
RESULTS = OUTPUT / "fold0_results.json"
CV_RESULTS = OUTPUT / "cv_results.json"
CALIBRATION = OUTPUT / "calibration.json"
OOF_FILE = OUTPUT / "oof_predictions.npz"
BASE_DIR = config.OUTPUT_DIR / "factor_cross_encoder_v2"
ACCEPTED = config.OUTPUT_DIR / "factor_cross_encoder_v2" / "oof_predictions.npz"
OLD_CROSS = config.OUTPUT_DIR / "factor_cross_encoder" / "oof_predictions.npz"
TRAINING_VERSION = "prototype-retrieval-contrast-v41"
EPOCHS = 1
REPLACEMENT_GRID = (0., .10, .20, .25, .35, .50, .75, 1.)


def _fold_paths(fold):
    return OUTPUT / f"fold{fold}_model.pt", OUTPUT / f"fold{fold}_valid.npz"


def _confusions(label):
    result = set()
    for group in CONFUSION_GROUPS:
        if label in group:
            result.update(item for item in group if item != label)
    return sorted(result)


def _retrieval_for_queries(texts, targets, groups, matrix):
    examples = [[""] * config.NUM_FACTORS for _ in texts]
    for label in range(config.NUM_FACTORS):
        positives = np.flatnonzero(targets[:, label] > 0)
        if not len(positives):
            continue
        similarity = (matrix @ matrix[positives].T).toarray()
        same_user = groups[:, None] == groups[positives][None, :]
        similarity[same_user] = -np.inf
        choice = similarity.argmax(1)
        invalid = ~np.isfinite(similarity[np.arange(len(texts)), choice])
        if invalid.any():
            fallback = np.resize(positives, int(invalid.sum()))
            selected = positives[choice]
            selected[invalid] = fallback
        else:
            selected = positives[choice]
        prototype = FACTOR_PROTOTYPES[label][0]
        for row, reference in enumerate(selected):
            examples[row][label] = _best_excerpt(texts[int(reference)], texts[row], prototype)
    return examples


class ContrastDataset(Dataset):
    def __init__(self, texts, targets, counts, examples, seed):
        rng = random.Random(seed); supports = targets.sum(0).clip(min=1)
        maximum = float(supports.max()); rows = []
        for row, truth in enumerate(targets):
            for label in np.flatnonzero(truth):
                candidates = [item for item in _confusions(int(label)) if not truth[item]]
                if not candidates:
                    continue
                # More frequent confusing labels provide stable negative supervision.
                negative = max(candidates, key=lambda item: int(supports[item]))
                prototype = rng.randrange(len(FACTOR_PROTOTYPES[int(label)]))
                neg_prototype = rng.randrange(len(FACTOR_PROTOTYPES[int(negative)]))
                tail = min(4., float((maximum / supports[int(label)]) ** .35))
                repeat = 1. + .15 * np.log1p(max(0., float(counts[row, label])-1.))
                rows.append((row, int(label), prototype, int(negative), neg_prototype,
                             min(5., tail*repeat)))
        rng.shuffle(rows)
        self.texts, self.examples, self.rows = texts, examples, rows

    def __len__(self): return len(self.rows)

    def __getitem__(self, index):
        row, label, proto, negative, neg_proto, weight = self.rows[index]
        text = self.texts[row]
        positive_hypothesis = _guided_hypothesis(
            label, FACTOR_PROTOTYPES[label][proto], self.examples[row][label], negative,
        )
        negative_hypothesis = _guided_hypothesis(
            negative, FACTOR_PROTOTYPES[negative][neg_proto],
            self.examples[row][negative], label,
        )
        return text, positive_hypothesis, negative_hypothesis, weight


def _collator(tokenizer):
    def collate(rows):
        texts, positive, negative, weights = zip(*rows)
        premises, hypotheses, owners, signs = [], [], [], []
        for owner, (text, pos, neg) in enumerate(zip(texts, positive, negative)):
            premises.extend((text, text)); hypotheses.extend((pos, neg))
            owners.extend((owner, owner)); signs.extend((1, 0))
        encoded = tokenizer(
            premises, hypotheses, padding=True, truncation="only_first",
            max_length=config.FACTOR_NLI_MAX_LENGTH,
            stride=min(128, config.FACTOR_NLI_MAX_LENGTH//3),
            return_overflowing_tokens=True, return_tensors="pt",
        )
        mapping = encoded.pop("overflow_to_sample_mapping").cpu().numpy()
        selected = []
        for pair, sign in enumerate(signs):
            indices = np.flatnonzero(mapping == pair)
            maximum = 2 if sign else 1
            if len(indices) > maximum:
                positions = np.linspace(0, len(indices)-1, maximum).round().astype(int)
                indices = indices[positions]
            selected.extend(indices.tolist())
        selected = np.asarray(selected, dtype=np.int64)
        take = torch.tensor(selected, dtype=torch.long)
        result = {key: value.index_select(0, take) for key, value in encoded.items()}
        result["pair_mapping"] = torch.tensor(mapping[selected], dtype=torch.long)
        result["owners"] = torch.tensor(owners, dtype=torch.long)
        result["signs"] = torch.tensor(signs, dtype=torch.float32)
        result["weights"] = torch.tensor(weights, dtype=torch.float32)
        return result
    return collate


def _train(model, loader, device):
    model.gradient_checkpointing_enable(); model.config.use_cache = False
    optimizer = AdamW(model.parameters(), lr=1.5e-6, weight_decay=config.WEIGHT_DECAY)
    accumulation = 16
    updates = max(1, int(np.ceil(len(loader)/accumulation))*EPOCHS)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, max(1, int(.08*updates)), updates,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=config.FP16)
    entailment = _entailment_index(model); not_entailment = 1-entailment
    model.train(); optimizer.zero_grad(set_to_none=True); losses=[]
    for step, batch in enumerate(tqdm(loader, desc="V41 contrast epoch 1"), 1):
        mapping=batch.pop("pair_mapping").to(device)
        owners=batch.pop("owners").to(device); signs=batch.pop("signs").to(device)
        weights=batch.pop("weights").to(device)
        inputs={key:value.to(device) for key,value in batch.items()}
        with torch.autocast(device_type=device.type, enabled=config.FP16):
            logits=model(**inputs).logits
            chunk=logits[:,entailment]-logits[:,not_entailment]
            margins=torch.stack([chunk[mapping==pair].max() for pair in range(len(signs))])
            raw=F.binary_cross_entropy_with_logits(margins,signs,reduction="none")
            pair_weight=weights[owners]
            classification=(raw*pair_weight).mean()
            contrasts=[]
            for owner in range(len(weights)):
                pos=margins[(owners==owner)&(signs>0.5)][0]
                neg=margins[(owners==owner)&(signs<0.5)][0]
                contrasts.append(F.relu(.35-pos+neg))
            contrast=torch.stack(contrasts).mean()
            loss=(classification+.20*contrast)/accumulation
        scaler.scale(loss).backward(); losses.append(float(loss.detach())*accumulation)
        if step%accumulation==0 or step==len(loader):
            scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            old=scaler.get_scale(); scaler.step(optimizer); scaler.update()
            if scaler.get_scale()>=old: scheduler.step()
            optimizer.zero_grad(set_to_none=True)
    return float(np.mean(losses))


def _train_fold(fold, train_idx, valid_idx, frame, targets, counts, all_groups,
                tokenizer, device):
    checkpoint, prediction_file = _fold_paths(fold)
    if checkpoint.exists() and prediction_file.exists():
        saved = np.load(prediction_file)
        if (np.array_equal(saved["valid_indices"], valid_idx)
                and saved["probabilities"].shape == (len(valid_idx), config.NUM_FACTORS)):
            print(f"V41 fold {fold}: resumed cached checkpoint and predictions", flush=True)
            return saved["probabilities"].astype(np.float32), {
                "fold": fold, "resumed": True, "valid_size": len(valid_idx),
            }
    seed_everything(config.SEED + 4141 + fold)
    train_texts=frame.text.iloc[train_idx].astype(str).tolist()
    valid_texts=frame.text.iloc[valid_idx].astype(str).tolist()
    vectorizer=_vectorizer(); train_matrix=vectorizer.fit_transform(train_texts)
    valid_matrix=vectorizer.transform(valid_texts)
    weights,diagnostics=_prototype_weights(vectorizer,train_matrix,targets[train_idx])
    train_examples=_retrieval_for_queries(
        train_texts,targets[train_idx],all_groups[train_idx],train_matrix,
    )
    # Outer validation retrieves only from outer training users.
    valid_examples=[[""]*config.NUM_FACTORS for _ in valid_texts]
    for label in range(config.NUM_FACTORS):
        positives=np.flatnonzero(targets[train_idx,label]>0)
        if not len(positives): continue
        similarity=valid_matrix@train_matrix[positives].T
        chosen=positives[np.asarray(similarity.argmax(axis=1)).ravel()]
        for row,reference in enumerate(chosen):
            valid_examples[row][label]=_best_excerpt(
                train_texts[int(reference)],valid_texts[row],FACTOR_PROTOTYPES[label][0],
            )
    confusion=[]
    for label in range(config.NUM_FACTORS):
        candidates=_confusions(label)
        confusion.append(max(candidates,key=lambda item:int(targets[train_idx,item].sum()))
                         if candidates else (label+1)%config.NUM_FACTORS)

    model=AutoModelForSequenceClassification.from_pretrained(
        config.FACTOR_NLI_MODEL_NAME,dtype=torch.float32,local_files_only=True,
    ).to(device)
    base_checkpoint = BASE_DIR / f"fold{fold}_model.pt"
    if not base_checkpoint.exists():
        raise FileNotFoundError(f"Missing accepted prototype fold: {base_checkpoint}")
    model.load_state_dict(torch.load(base_checkpoint,map_location=device))
    dataset=ContrastDataset(
        train_texts,targets[train_idx],counts[train_idx],train_examples,
        config.SEED+4141+fold,
    )
    loader=DataLoader(dataset,batch_size=1,shuffle=True,collate_fn=_collator(tokenizer),num_workers=0)
    loss=_train(model,loader,device); torch.save(model.state_dict(),checkpoint)
    candidate=_predict(model,tokenizer,valid_texts,valid_examples,weights,confusion,device)
    np.savez_compressed(
        prediction_file, valid_indices=valid_idx, probabilities=candidate,
        training_version=TRAINING_VERSION,
    )
    summary = {
        "fold": fold, "resumed": False, "valid_size": len(valid_idx),
        "pair_count": len(dataset), "train_loss": loss,
        "prototype_weight_diagnostics": diagnostics,
    }
    del model, loader, dataset
    if device.type == "cuda": torch.cuda.empty_cache()
    return candidate, summary


def _component_probability(base, old, accepted, candidate, calibration, indices,
                           replacement):
    prototype = (1-replacement)*accepted[indices] + replacement*candidate[indices]
    return (
        float(calibration["base_weight"])*base[indices]
        + float(calibration["old_cross_weight"])*old[indices]
        + float(calibration["new_cross_weight"])*prototype
    )


def _fold_score(targets, probability, valid_idx, prevalence, ratio):
    return float(f1_score(
        targets[valid_idx], _rank_decode(probability, prevalence, ratio),
        average="macro", zero_division=0,
    ))


def main(only_fold0=True):
    if not torch.cuda.is_available(): raise RuntimeError("V41 requires CUDA")
    OUTPUT.mkdir(parents=True,exist_ok=True); seed_everything(config.SEED+4141)
    frame=load_train_data().reset_index(drop=True)
    targets=np.vstack(frame.factor_vector.to_numpy()).astype(np.int8)
    counts=np.vstack(frame.factor_counts.to_numpy()).astype(np.float32)
    all_groups=frame.anon_user_id.astype(str).to_numpy()
    folds=list(StratifiedGroupKFold(n_splits=config.N_FOLDS,shuffle=True,random_state=config.SEED)
               .split(np.zeros(len(frame)),frame.risk_label,all_groups))
    tokenizer=AutoTokenizer.from_pretrained(
        config.FACTOR_NLI_MODEL_NAME,use_fast=True,local_files_only=True,
    )
    device=torch.device(config.DEVICE)
    candidate_oof=np.zeros((len(frame),config.NUM_FACTORS),dtype=np.float32)
    summaries=[]
    selected=folds[:1] if only_fold0 else folds
    for fold,(train_idx,valid_idx) in enumerate(selected):
        probability,summary=_train_fold(
            fold,train_idx,valid_idx,frame,targets,counts,all_groups,tokenizer,device,
        )
        candidate_oof[valid_idx]=probability; summaries.append(summary)

    current,all_targets,calibration=_current_components()
    accepted=np.load(ACCEPTED)["probabilities"].astype(np.float32)
    old=np.load(OLD_CROSS)["probabilities"].astype(np.float32)
    base=(current-float(calibration["old_cross_weight"])*old
          -float(calibration["new_cross_weight"])*accepted)/float(calibration["base_weight"])
    ratio=float(calibration["prevalence_ratio"])

    if only_fold0:
        train_idx,valid_idx=folds[0]; prevalence=all_targets[train_idx].mean(0)
        baseline=_fold_score(all_targets[valid_idx],current[valid_idx],valid_idx=np.arange(len(valid_idx)),
                             prevalence=prevalence,ratio=ratio)
        grid=[]
        for replacement in REPLACEMENT_GRID:
            probability=_component_probability(
                base,old,accepted,candidate_oof,calibration,valid_idx,replacement,
            )
            score=_fold_score(all_targets[valid_idx],probability,np.arange(len(valid_idx)),prevalence,ratio)
            grid.append({"replacement":replacement,"macro_f1":score,"delta":score-baseline})
        fixed=next(x for x in grid if x["replacement"]==.25)
        payload={"training_version":TRAINING_VERSION,"evaluation_scope":"strict user-disjoint fold0",
                 "folds":summaries,"current_baseline_macro_f1":baseline,
                 "fixed_25pct_replacement":fixed,
                 "grid":sorted(grid,key=lambda x:x["macro_f1"],reverse=True),
                 "promising":bool(fixed["delta"]>=.003)}
        RESULTS.write_text(json.dumps(payload,indent=2),encoding="utf-8")
        print(json.dumps(payload,indent=2),flush=True); return payload

    # Every row now has a prediction from a model that never saw its user.
    baseline_predictions=np.zeros_like(all_targets,dtype=bool)
    fixed_predictions=np.zeros_like(all_targets,dtype=bool)
    nested_predictions=np.zeros_like(all_targets,dtype=bool)
    fold_grids=[]; nested_parameters=[]
    for held,(held_train,held_valid) in enumerate(folds):
        prevalence=all_targets[held_train].mean(0)
        baseline_predictions[held_valid]=_rank_decode(current[held_valid],prevalence,ratio)
        scores=[]
        for replacement in REPLACEMENT_GRID:
            other_scores=[]
            for other,(other_train,other_valid) in enumerate(folds):
                if other==held: continue
                probability=_component_probability(
                    base,old,accepted,candidate_oof,calibration,other_valid,replacement,
                )
                other_scores.append(_fold_score(
                    all_targets[other_valid],probability,np.arange(len(other_valid)),
                    all_targets[other_train].mean(0),ratio,
                ))
            scores.append({"replacement":replacement,
                           "mean_selection_f1":float(np.mean(other_scores))})
        selected_parameter=max(scores,key=lambda x:(x["mean_selection_f1"],-x["replacement"]))
        replacement=float(selected_parameter["replacement"])
        probability=_component_probability(
            base,old,accepted,candidate_oof,calibration,held_valid,replacement,
        )
        nested_predictions[held_valid]=_rank_decode(probability,prevalence,ratio)
        nested_parameters.append({"held_fold":held,**selected_parameter})
        fold_grids.append({"held_fold":held,"selection_grid":scores})

    production_weight=float(np.median([x["replacement"] for x in nested_parameters]))
    for fold,(train_idx,valid_idx) in enumerate(folds):
        probability=_component_probability(
            base,old,accepted,candidate_oof,calibration,valid_idx,production_weight,
        )
        fixed_predictions[valid_idx]=_rank_decode(
            probability,all_targets[train_idx].mean(0),ratio,
        )
    baseline_score=float(f1_score(all_targets,baseline_predictions,average="macro",zero_division=0))
    nested_score=float(f1_score(all_targets,nested_predictions,average="macro",zero_division=0))
    production_score=float(f1_score(all_targets,fixed_predictions,average="macro",zero_division=0))
    adopted=bool(nested_score>=baseline_score+.003 and production_score>=baseline_score+.003)
    payload={
        "training_version":TRAINING_VERSION,
        "evaluation_scope":"five-fold nested user-disjoint OOF",
        "folds":summaries,"baseline_macro_f1":baseline_score,
        "nested_crossfit_macro_f1":nested_score,"nested_delta":nested_score-baseline_score,
        "production_replacement":production_weight,
        "production_macro_f1":production_score,"production_delta":production_score-baseline_score,
        "nested_parameters":nested_parameters,"selection_details":fold_grids,
        "adoption_gate":"nested and production must both improve by at least 0.003",
        "adopted":adopted,
    }
    np.savez_compressed(
        OOF_FILE,probabilities=candidate_oof,targets=all_targets,
        nested_predictions=nested_predictions,production_predictions=fixed_predictions,
    )
    CV_RESULTS.write_text(json.dumps(payload,indent=2),encoding="utf-8")
    CALIBRATION.write_text(json.dumps({
        "training_version":TRAINING_VERSION,"replacement":production_weight,
        "prevalence_ratio":ratio,"nested_crossfit_macro_f1":nested_score,
        "baseline_macro_f1":baseline_score,"production_macro_f1":production_score,
        "adopted":adopted,
    },indent=2),encoding="utf-8")
    print(json.dumps(payload,indent=2),flush=True); return payload


if __name__=="__main__": main()
