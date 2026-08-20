"""Four-post causal-window ablation of the leaderboard-positive V58 expert."""
from __future__ import annotations

import json

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold

from baseline import _post_phrase_f1
from configs.config import config
from inference.task1_evidence_v4 import correct_risk_only
from preprocess.preprocess import load_train_data
from trainer.task1_factor_trajectory_v58 import (
    _current_baseline, _fold0_factor_probabilities, _outer_split, _sequences,
)
from trainer.task1_local_counterfactual_train_v56 import _decode
from utils.seed import seed_everything
from utils.task1_metric import task1_score


OUTPUT = config.OUTPUT_DIR / "task1_windowed_trajectory_v61"
RESULTS = OUTPUT / "results.json"
SEEDS = (6161, 16161, 26161)
WINDOW = 4
FIXED_WEIGHT = .05


class WindowedTrajectory(nn.Module):
    def __init__(self):
        super().__init__()
        self.risk = nn.Sequential(nn.Linear(19,32),nn.GELU(),nn.LayerNorm(32))
        self.protective = nn.Sequential(nn.Linear(5,16),nn.GELU(),nn.LayerNorm(16))
        self.gru = nn.GRU(48,40,batch_first=True)
        self.head = nn.Sequential(nn.Linear(88,48),nn.GELU(),nn.Dropout(.15),nn.Linear(48,4))
    def forward(self,factors):
        current=torch.cat((self.risk(factors[...,:19]),self.protective(factors[...,19:])),dim=-1)
        states=[]
        for step in range(current.size(1)):
            start=max(0,step-WINDOW+1)
            hidden,_=self.gru(current[:,start:step+1])
            states.append(hidden[:,-1])
        history=torch.stack(states,dim=1)
        return self.head(torch.cat((history,current),dim=-1))


@torch.no_grad()
def _infer(model,sequences,size,device):
    model.eval(); result=np.zeros((size,4),dtype=np.float32)
    for idx,x,_ in sequences:
        result[idx]=torch.softmax(model(x.unsqueeze(0).to(device))[0],-1).cpu().numpy()
    return result


def _train(frame,factors,labels,fit_idx,valid_idx,seed,epochs=None):
    seed_everything(seed); device=torch.device(config.DEVICE); model=WindowedTrajectory().to(device)
    count=np.bincount(labels[fit_idx],minlength=4).astype(np.float32)
    weight=np.sqrt(count.sum()/np.maximum(count,1)); weight/=weight.mean()
    loss_fn=nn.CrossEntropyLoss(weight=torch.tensor(weight,device=device))
    optimizer=torch.optim.AdamW(model.parameters(),lr=8e-4,weight_decay=1e-3)
    train=_sequences(frame,fit_idx,factors,labels); valid=_sequences(frame,valid_idx,factors,labels)
    rng=np.random.default_rng(seed); best={"score":-1.,"epoch":1,"state":None}; stale=0
    for epoch in range(1,int(epochs or 45)+1):
        model.train(); losses=[]
        for pos in rng.permutation(len(train)):
            _,x,y=train[int(pos)]; x=x.to(device); y=y.to(device)
            if epochs is None: x=x*(torch.rand_like(x)>.04).float()
            optimizer.zero_grad(set_to_none=True); logits=model(x.unsqueeze(0))[0]
            loss=loss_fn(logits,y); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step(); losses.append(float(loss.detach()))
        if epochs is not None: continue
        probability=_infer(model,valid,len(frame),device)
        score=float(f1_score(labels[valid_idx],probability[valid_idx].argmax(1),average="weighted",zero_division=0))
        if score>best["score"]+1e-5:
            best={"score":score,"epoch":epoch,"loss":float(np.mean(losses)),
                  "state":{k:v.detach().cpu().clone() for k,v in model.state_dict().items()}}; stale=0
        else: stale+=1
        if stale>=10: break
    if epochs is None: model.load_state_dict(best.pop("state"))
    return model,best


def main():
    OUTPUT.mkdir(parents=True,exist_ok=True); frame=load_train_data().reset_index(drop=True)
    labels=frame.risk_label.to_numpy(dtype=np.int64); groups=frame.anon_user_id.astype(str).to_numpy()
    train_idx,valid_idx=_outer_split(frame); factors=_fold0_factor_probabilities(frame,train_idx,valid_idx)
    a,b=next(StratifiedGroupKFold(n_splits=4,shuffle=True,random_state=config.SEED+6161).split(
        np.zeros(len(train_idx)),labels[train_idx],groups[train_idx]))
    inner=[]
    for seed in SEEDS:
        model,row=_train(frame,factors,labels,train_idx[a],train_idx[b],seed)
        inner.append({"seed":seed,**row}); del model
    epochs=max(1,int(round(np.median([x["epoch"] for x in inner]))))
    seq=_sequences(frame,valid_idx,factors,labels); values=[]
    for seed in SEEDS:
        model,_=_train(frame,factors,labels,train_idx,valid_idx,seed,epochs)
        values.append(_infer(model,seq,len(frame),torch.device(config.DEVICE))[valid_idx]); del model
    expert=np.mean(values,axis=0)
    baseline_probability,records,starts,ends,parameters=_current_baseline(frame,train_idx,valid_idx)
    texts=frame.text.iloc[valid_idx].astype(str).tolist(); truth=labels[valid_idx]
    gold=[list(frame.iloc[int(i)].evidence) for i in valid_idx]; rows=[]; stored={}
    for weight in (0.,.02,.05,.08,.10):
        p=(1-weight)*baseline_probability+weight*expert
        risk=np.asarray([correct_risk_only(t,int(y)) for t,y in zip(texts,p.argmax(1))])
        evidence=_decode(records,risk,starts,ends,parameters)
        phrase=np.asarray([_post_phrase_f1(a,b) for a,b in zip(evidence,gold)])
        rf=float(f1_score(truth,risk,average="weighted",zero_division=0)); pf=float(phrase.mean())
        rows.append({"weight":weight,"risk_f1":rf,"phrase_f1":pf,"task1":task1_score(rf,pf)})
        stored[weight]=(risk,phrase)
    baseline=rows[0]; fixed=next(x for x in rows if x["weight"]==FIXED_WEIGHT)
    old_risk,old_phrase=stored[0.]; new_risk,new_phrase=stored[FIXED_WEIGHT]
    unique=np.unique(groups[valid_idx]); rng=np.random.default_rng(616161); deltas=[]
    for _ in range(4000):
        sample=rng.choice(unique,len(unique),replace=True)
        pos=np.concatenate([np.flatnonzero(groups[valid_idx]==u) for u in sample])
        old=f1_score(truth[pos],old_risk[pos],average="weighted",zero_division=0)
        new=f1_score(truth[pos],new_risk[pos],average="weighted",zero_division=0)
        deltas.append(task1_score(new,new_phrase[pos].mean())-task1_score(old,old_phrase[pos].mean()))
    deltas=np.asarray(deltas); bootstrap={"mean_delta":float(deltas.mean()),
        "p05_delta":float(np.quantile(deltas,.05)),"p95_delta":float(np.quantile(deltas,.95)),
        "positive_fraction":float((deltas>0).mean())}
    adopted=bool(fixed["task1"]>=baseline["task1"]+.003 and bootstrap["positive_fraction"]>=.8
                 and bootstrap["p05_delta"]>=0)
    payload={"training_version":"task1-four-post-factor-trajectory-v61",
        "evaluation_scope":"outer user-disjoint fold0; current-post target",
        "method":{"window":WINDOW,"seeds":list(SEEDS),"inner_selected_epochs":epochs,
                  "fixed_weight":FIXED_WEIGHT},"inner_selection":inner,
        "standalone_risk_f1":float(f1_score(truth,expert.argmax(1),average="weighted",zero_division=0)),
        "baseline":baseline,"fixed_candidate":{**fixed,"changed_risk":int((new_risk!=old_risk).sum()),
            "confusion":confusion_matrix(truth,new_risk,labels=np.arange(4)).tolist()},
        "weight_ablation_outer_diagnostic_only":rows,"user_cluster_bootstrap":bootstrap,"adopted":adopted}
    RESULTS.write_text(json.dumps(payload,indent=2),encoding="utf8"); print(json.dumps(payload,indent=2),flush=True)
    return payload

if __name__=="__main__": main()
