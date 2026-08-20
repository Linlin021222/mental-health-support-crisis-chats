"""Fold-0 protective-factor specialist following the PFA two-branch design."""
from __future__ import annotations

import json
import random

import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_cosine_schedule_with_warmup

from configs.config import config
from inference.factor_nli import _entailment_index, _rank_decode
from preprocess.factor_prototypes import FACTOR_PROTOTYPES
from preprocess.preprocess import load_train_data
from trainer.factor_cross_encoder_v2 import PrototypePairDataset, _collator
from trainer.factor_mhlat_v4 import _current_components
from utils.seed import seed_everything


OUTPUT=config.OUTPUT_DIR/"factor_protective_branch_v43"
CHECKPOINT=OUTPUT/"fold0_model.pt"
PREDICTIONS=OUTPUT/"fold0_valid.npz"
RESULTS=OUTPUT/"fold0_results.json"
BASE_CHECKPOINT=config.OUTPUT_DIR/"factor_cross_encoder_v2"/"fold0_model.pt"
ACCEPTED=config.OUTPUT_DIR/"factor_cross_encoder_v2"/"oof_predictions.npz"
OLD=config.OUTPUT_DIR/"factor_cross_encoder"/"oof_predictions.npz"
TRAINING_VERSION="pfa-protective-specialist-v43"
PROTECTIVE=tuple(range(19,24))


def _pairs(targets,counts,seed):
    rng=random.Random(seed); supports=targets.sum(0).clip(min=1); maximum=float(supports[19:].max())
    pairs=[]
    for row,truth in enumerate(targets):
        positives=[label for label in PROTECTIVE if truth[label]]
        negatives=[label for label in PROTECTIVE if not truth[label]]
        for label in positives:
            tail=min(5.,float((maximum/supports[label])**.4))
            repeat=min(1.5,1.+.2*np.log1p(max(0.,float(counts[row,label])-1.)))
            for prototype in range(len(FACTOR_PROTOTYPES[label])):
                pairs.append((row,label,prototype,1,min(6.,tail*repeat)))
        # All semantic boundary labels when a protective concept is present;
        # otherwise two rotating negatives prevent empty posts dominating.
        chosen=negatives if positives else rng.sample(negatives,min(2,len(negatives)))
        for label in chosen:
            pairs.append((row,label,rng.randrange(len(FACTOR_PROTOTYPES[label])),0,1.))
    rng.shuffle(pairs); return pairs


def _train(model,loader,device):
    model.gradient_checkpointing_enable(); model.config.use_cache=False
    optimizer=AdamW(model.parameters(),lr=1.5e-6,weight_decay=config.WEIGHT_DECAY)
    accumulation=16; updates=max(1,int(np.ceil(len(loader)/accumulation)))
    scheduler=get_cosine_schedule_with_warmup(optimizer,max(1,int(.08*updates)),updates)
    scaler=torch.amp.GradScaler("cuda",enabled=config.FP16)
    entailment=_entailment_index(model); other=1-entailment
    model.train(); optimizer.zero_grad(set_to_none=True); losses=[]
    for step,batch in enumerate(tqdm(loader,desc="V43 protective branch"),1):
        binary=batch.pop("targets").to(device).float(); weights=batch.pop("weights").to(device)
        mapping=batch.pop("pair_mapping").to(device); inputs={k:v.to(device) for k,v in batch.items()}
        with torch.autocast(device_type=device.type,enabled=config.FP16):
            logits=model(**inputs).logits; margin=logits[:,entailment]-logits[:,other]
            pair=torch.stack([margin[mapping==index].max() for index in range(len(binary))])
            raw=torch.nn.functional.binary_cross_entropy_with_logits(pair,binary,reduction="none")
            loss=(raw*weights).mean()/accumulation
        scaler.scale(loss).backward(); losses.append(float(loss.detach())*accumulation)
        if step%accumulation==0 or step==len(loader):
            scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            old=scaler.get_scale(); scaler.step(optimizer); scaler.update()
            if scaler.get_scale()>=old: scheduler.step()
            optimizer.zero_grad(set_to_none=True)
    return float(np.mean(losses))


@torch.no_grad()
def _predict(model,tokenizer,texts,device):
    model.eval(); entailment=_entailment_index(model); result=np.zeros((len(texts),config.NUM_FACTORS),np.float32)
    batch_size=max(1,config.FACTOR_CROSS_ENCODER_BATCH_SIZE)
    for label in tqdm(PROTECTIVE,desc="V43 protective labels"):
        scores=np.zeros((len(texts),len(FACTOR_PROTOTYPES[label])),np.float32)
        for prototype,hypothesis in enumerate(FACTOR_PROTOTYPES[label]):
            for start in range(0,len(texts),batch_size):
                batch=texts[start:start+batch_size]
                encoded=tokenizer(batch,[hypothesis]*len(batch),padding=True,truncation="only_first",
                                  max_length=config.FACTOR_NLI_MAX_LENGTH,stride=128,
                                  return_overflowing_tokens=True,return_tensors="pt")
                mapping=encoded.pop("overflow_to_sample_mapping").cpu().numpy(); selected=[]
                for local in range(len(batch)):
                    indices=np.flatnonzero(mapping==local)
                    if len(indices)>config.FACTOR_NLI_MAX_CHUNKS:
                        positions=np.linspace(0,len(indices)-1,config.FACTOR_NLI_MAX_CHUNKS).round().astype(int)
                        indices=indices[positions]
                    selected.extend(indices.tolist())
                selected=np.asarray(selected); local_map=mapping[selected]; take=torch.tensor(selected,dtype=torch.long)
                inputs={k:v.index_select(0,take).to(device) for k,v in encoded.items()}
                with torch.autocast(device_type=device.type,enabled=True):
                    values=torch.softmax(model(**inputs).logits.float(),-1)[:,entailment].cpu().numpy()
                for local in range(len(batch)):
                    scores[start+local,prototype]=values[local_map==local].max()
        result[:,label]=scores.mean(1)
    return result


def main():
    if not torch.cuda.is_available(): raise RuntimeError("V43 requires CUDA")
    OUTPUT.mkdir(parents=True,exist_ok=True); seed_everything(config.SEED+4343)
    frame=load_train_data().reset_index(drop=True); targets=np.vstack(frame.factor_vector).astype(np.int8)
    counts=np.vstack(frame.factor_counts).astype(np.float32); groups=frame.anon_user_id.astype(str).to_numpy()
    folds=list(StratifiedGroupKFold(5,shuffle=True,random_state=config.SEED).split(np.zeros(len(frame)),frame.risk_label,groups))
    train_idx,valid_idx=folds[0]; train_texts=frame.text.iloc[train_idx].astype(str).tolist(); valid_texts=frame.text.iloc[valid_idx].astype(str).tolist()
    tokenizer=AutoTokenizer.from_pretrained(config.FACTOR_NLI_MODEL_NAME,use_fast=True,local_files_only=True)
    device=torch.device(config.DEVICE); model=AutoModelForSequenceClassification.from_pretrained(
        config.FACTOR_NLI_MODEL_NAME,dtype=torch.float32,local_files_only=True).to(device)
    model.load_state_dict(torch.load(BASE_CHECKPOINT,map_location=device))
    pairs=_pairs(targets[train_idx],counts[train_idx],config.SEED+4343)
    loader=DataLoader(PrototypePairDataset(train_texts,pairs),batch_size=1,shuffle=True,
                      collate_fn=_collator(tokenizer),num_workers=0)
    loss=_train(model,loader,device); torch.save(model.state_dict(),CHECKPOINT)
    candidate=_predict(model,tokenizer,valid_texts,device)
    current,all_targets,cal=_current_components(); accepted=np.load(ACCEPTED)["probabilities"].astype(np.float32)
    old=np.load(OLD)["probabilities"].astype(np.float32); prevalence=all_targets[train_idx].mean(0); ratio=float(cal["prevalence_ratio"])
    baseline=float(f1_score(all_targets[valid_idx],_rank_decode(current[valid_idx],prevalence,ratio),average="macro",zero_division=0))
    grid=[]
    for replacement in (0.,.10,.20,.35,.50,.75,1.):
        probability=current[valid_idx].copy()
        probability[:,19:]+=float(cal["new_cross_weight"])*replacement*(candidate[:,19:]-accepted[valid_idx,19:])
        prediction=_rank_decode(probability,prevalence,ratio)
        macro=float(f1_score(all_targets[valid_idx],prediction,average="macro",zero_division=0))
        protective=float(f1_score(all_targets[valid_idx,19:],prediction[:,19:],average="macro",zero_division=0))
        grid.append({"replacement":replacement,"macro_f1":macro,"protective_macro_f1":protective,"delta":macro-baseline})
    fixed=next(x for x in grid if x["replacement"]==.20)
    payload={"training_version":TRAINING_VERSION,"evaluation_scope":"strict user-disjoint fold0",
             "paper_transfer":"independent protective-factor expert branch","pair_count":len(pairs),"train_loss":loss,
             "baseline_macro_f1":baseline,"fixed_20pct":fixed,"grid":sorted(grid,key=lambda x:x["macro_f1"],reverse=True),
             "promising":bool(fixed["delta"]>=.003)}
    np.savez_compressed(PREDICTIONS,valid_indices=valid_idx,probabilities=candidate)
    RESULTS.write_text(json.dumps(payload,indent=2),encoding="utf-8"); print(json.dumps(payload,indent=2),flush=True); return payload


if __name__=="__main__": main()
