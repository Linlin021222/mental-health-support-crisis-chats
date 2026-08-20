"""Strict fold-0 continuation on paper-aligned factor boundaries."""
from __future__ import annotations

import json, random
import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_cosine_schedule_with_warmup

from configs.config import config
from inference.factor_nli import _entailment_index, _rank_decode
from preprocess.factor_paper_boundary_prototypes_v44 import PAPER_BOUNDARY_PROTOTYPES as BANK
from preprocess.preprocess import load_train_data
from trainer.factor_cross_encoder_v2 import CONFUSION_GROUPS
from trainer.factor_mhlat_v4 import _current_components
from utils.seed import seed_everything

OUTPUT=config.OUTPUT_DIR/"factor_paper_boundary_cross_v44"; CHECKPOINT=OUTPUT/"fold0_model.pt"
PREDICTIONS=OUTPUT/"fold0_valid.npz"; RESULTS=OUTPUT/"fold0_results.json"
BASE=config.OUTPUT_DIR/"factor_cross_encoder_v2"/"fold0_model.pt"
ACCEPTED=config.OUTPUT_DIR/"factor_cross_encoder_v2"/"oof_predictions.npz"
OLD=config.OUTPUT_DIR/"factor_cross_encoder"/"oof_predictions.npz"
HARD=config.OUTPUT_DIR/"factor_cross_encoder_v2"/"fold0_train_hard.npz"
TRAINING_VERSION="paper-table3-boundary-cross-v44"
TARGETS=(0,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23)

class PairData(Dataset):
    def __init__(self,texts,pairs): self.texts,self.pairs=texts,pairs
    def __len__(self): return len(self.pairs)
    def __getitem__(self,index):
        row,label,prototype,target,weight=self.pairs[index]
        return self.texts[row],label,prototype,target,weight

def _pairs(targets,counts,hard,seed):
    rng=random.Random(seed); supports=targets.sum(0).clip(min=1); maximum=float(supports.max()); pairs=[]
    for row,truth in enumerate(targets):
        positives=[x for x in TARGETS if truth[x]]; false=[x for x in TARGETS if not truth[x]]
        confusion=set()
        for positive in positives:
            for group in CONFUSION_GROUPS:
                if positive in group: confusion.update(x for x in group if x in TARGETS and not truth[x])
        ranked=sorted(false,key=lambda x:float(hard[row,x]),reverse=True)
        negatives=list(dict.fromkeys(list(confusion)+ranked))[:max(6,2*len(positives))]
        for label in positives:
            support=int(supports[label]); selected=range(3) if support<80 else ((0,2) if support<250 else (rng.randrange(3),))
            tail=min(5.,float((maximum/supports[label])**.35)); repeat=min(1.5,1.+.2*np.log1p(max(0.,counts[row,label]-1.)))
            for prototype in selected: pairs.append((row,label,int(prototype),1,min(6.,tail*repeat)))
        for label in negatives:
            # Boundary prototype is deliberately over-sampled for absent confusing labels.
            prototype=2 if label in confusion else rng.randrange(3)
            pairs.append((row,label,prototype,0,1.3 if label in confusion else 1.))
    rng.shuffle(pairs); return pairs

def _collator(tokenizer):
    def collate(rows):
        texts,labels,prototypes,targets,weights=zip(*rows); hypotheses=[BANK[int(l)][int(p)] for l,p in zip(labels,prototypes)]
        encoded=tokenizer(list(texts),hypotheses,padding=True,truncation="only_first",max_length=config.FACTOR_NLI_MAX_LENGTH,
                          stride=128,return_overflowing_tokens=True,return_tensors="pt")
        mapping=encoded.pop("overflow_to_sample_mapping").cpu().numpy(); selected=[]
        for local,(label,prototype,target) in enumerate(zip(labels,prototypes,targets)):
            indices=np.flatnonzero(mapping==local); maximum=2 if int(target)>0 else 1
            if len(indices)>maximum:
                positions=np.linspace(0,len(indices)-1,maximum).round().astype(int); indices=indices[positions]
            selected.extend(indices.tolist())
        selected=np.asarray(selected); take=torch.tensor(selected,dtype=torch.long)
        result={k:v.index_select(0,take) for k,v in encoded.items()}; result["pair_mapping"]=torch.tensor(mapping[selected])
        result["targets"]=torch.tensor(targets,dtype=torch.float32); result["weights"]=torch.tensor(weights,dtype=torch.float32)
        return result
    return collate

def _train(model,loader,device):
    model.gradient_checkpointing_enable(); model.config.use_cache=False; optimizer=AdamW(model.parameters(),lr=1.5e-6,weight_decay=.01)
    # Four semantic pairs per micro-batch and four-way accumulation preserve the
    # original effective batch of 16 while avoiding the batch-size-one bottleneck.
    accumulation=4; updates=max(1,int(np.ceil(len(loader)/accumulation))); scheduler=get_cosine_schedule_with_warmup(optimizer,max(1,int(.08*updates)),updates)
    scaler=torch.amp.GradScaler("cuda",enabled=True); entailment=_entailment_index(model); other=1-entailment
    model.train(); optimizer.zero_grad(set_to_none=True); losses=[]
    for step,batch in enumerate(tqdm(loader,desc="V44 paper boundaries"),1):
        truth=batch.pop("targets").to(device); weights=batch.pop("weights").to(device); mapping=batch.pop("pair_mapping").to(device)
        inputs={k:v.to(device) for k,v in batch.items()}
        with torch.autocast("cuda",enabled=True):
            logits=model(**inputs).logits; margin=logits[:,entailment]-logits[:,other]
            pair=torch.stack([margin[mapping==i].max() for i in range(len(truth))])
            loss=(torch.nn.functional.binary_cross_entropy_with_logits(pair,truth,reduction="none")*weights).mean()/accumulation
        scaler.scale(loss).backward(); losses.append(float(loss.detach())*accumulation)
        if step%accumulation==0 or step==len(loader):
            scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); old=scaler.get_scale(); scaler.step(optimizer); scaler.update()
            if scaler.get_scale()>=old: scheduler.step()
            optimizer.zero_grad(set_to_none=True)
    return float(np.mean(losses))

@torch.no_grad()
def _predict(model,tokenizer,texts,device):
    model.eval(); entailment=_entailment_index(model); result=np.zeros((len(texts),24),np.float32); bs=config.FACTOR_CROSS_ENCODER_BATCH_SIZE
    for label in tqdm(TARGETS,desc="V44 boundary labels"):
        scores=np.zeros((len(texts),3),np.float32)
        for proto,hypothesis in enumerate(BANK[label]):
            for start in range(0,len(texts),bs):
                batch=texts[start:start+bs]; encoded=tokenizer(batch,[hypothesis]*len(batch),padding=True,truncation="only_first",
                    max_length=config.FACTOR_NLI_MAX_LENGTH,stride=128,return_overflowing_tokens=True,return_tensors="pt")
                mapping=encoded.pop("overflow_to_sample_mapping").cpu().numpy(); selected=[]
                for local in range(len(batch)):
                    idx=np.flatnonzero(mapping==local)
                    if len(idx)>config.FACTOR_NLI_MAX_CHUNKS: idx=idx[np.linspace(0,len(idx)-1,config.FACTOR_NLI_MAX_CHUNKS).round().astype(int)]
                    selected.extend(idx.tolist())
                selected=np.asarray(selected); local_map=mapping[selected]; take=torch.tensor(selected)
                inputs={k:v.index_select(0,take).to(device) for k,v in encoded.items()}
                with torch.autocast("cuda",enabled=True): value=torch.softmax(model(**inputs).logits.float(),-1)[:,entailment].cpu().numpy()
                for local in range(len(batch)): scores[start+local,proto]=value[local_map==local].max()
        # The third hypothesis is the complete definition plus its exclusion
        # rule, not an independently positive "X is insufficient" statement.
        result[:,label]=.40*scores[:,0]+.40*scores[:,1]+.20*scores[:,2]
    return result

def main():
    if not torch.cuda.is_available(): raise RuntimeError("V44 requires CUDA")
    OUTPUT.mkdir(parents=True,exist_ok=True); seed_everything(4444); frame=load_train_data().reset_index(drop=True)
    y=np.vstack(frame.factor_vector).astype(np.int8); counts=np.vstack(frame.factor_counts).astype(np.float32); groups=frame.anon_user_id.astype(str).to_numpy()
    train_idx,valid_idx=next(StratifiedGroupKFold(5,shuffle=True,random_state=config.SEED).split(np.zeros(len(frame)),frame.risk_label,groups))
    texts=frame.text.astype(str); hard=np.load(HARD)["probabilities"].astype(np.float32); pairs=_pairs(y[train_idx],counts[train_idx],hard,4444)
    tokenizer=AutoTokenizer.from_pretrained(config.FACTOR_NLI_MODEL_NAME,use_fast=True,local_files_only=True); device=torch.device("cuda")
    model=AutoModelForSequenceClassification.from_pretrained(config.FACTOR_NLI_MODEL_NAME,dtype=torch.float32,local_files_only=True).to(device)
    model.load_state_dict(torch.load(BASE,map_location=device)); loader=DataLoader(PairData(texts.iloc[train_idx].tolist(),pairs),batch_size=4,shuffle=True,collate_fn=_collator(tokenizer))
    loss=_train(model,loader,device); torch.save(model.state_dict(),CHECKPOINT); candidate=_predict(model,tokenizer,texts.iloc[valid_idx].tolist(),device)
    current,targets,cal=_current_components(); accepted=np.load(ACCEPTED)["probabilities"]; prevalence=targets[train_idx].mean(0); ratio=float(cal["prevalence_ratio"])
    baseline=float(f1_score(targets[valid_idx],_rank_decode(current[valid_idx],prevalence,ratio),average="macro",zero_division=0)); grid=[]
    columns=np.asarray(TARGETS)
    for weight in (0.,.10,.20,.25,.35,.50,.75,1.):
        probability=current[valid_idx].copy(); probability[:,columns]+=float(cal["new_cross_weight"])*weight*(candidate[:,columns]-accepted[valid_idx][:,columns])
        score=float(f1_score(targets[valid_idx],_rank_decode(probability,prevalence,ratio),average="macro",zero_division=0))
        grid.append({"replacement":weight,"macro_f1":score,"delta":score-baseline})
    fixed=next(x for x in grid if x["replacement"]==.20); payload={"training_version":TRAINING_VERSION,"evaluation_scope":"strict user-disjoint fold0",
        "target_labels":[config.ID2FACTOR[x] for x in TARGETS],"pair_count":len(pairs),"train_loss":loss,"baseline_macro_f1":baseline,
        "fixed_20pct":fixed,"grid":sorted(grid,key=lambda x:x["macro_f1"],reverse=True),"promising":bool(fixed["delta"]>=.003)}
    np.savez_compressed(PREDICTIONS,valid_indices=valid_idx,probabilities=candidate); RESULTS.write_text(json.dumps(payload,indent=2),encoding="utf-8")
    print(json.dumps(payload,indent=2),flush=True); return payload

if __name__=="__main__": main()
