"""Same-post factor-aware residual calibration for current Task-1 risk."""
from __future__ import annotations

import json
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from baseline import _post_phrase_f1
from configs.config import config
from preprocess.preprocess import load_train_data
from trainer.task1_atomic_v25 import _load_records
from trainer.task1_factor_bridge_v54 import _accepted_factor_oof
from trainer.task1_factor_trajectory_v58 import _current_baseline, _decode, _risk, _outer_split
from utils.task1_metric import task1_score


OUTPUT=config.OUTPUT_DIR/"task1_same_post_factor_v62"
RESULTS=OUTPUT/"results.json"
PREDICTIONS=OUTPUT/"strict_predictions.npz"
TRAINING_VERSION="task1-same-post-factor-residual-v62"


def _features(base,factors):
    base=np.asarray(base,np.float64); factors=np.asarray(factors,np.float64)
    log=np.log(np.clip(base,1e-6,1.)); risk=factors[:,:19]; protective=factors[:,19:]
    risk_sum=risk.mean(1,keepdims=True); protect_sum=protective.mean(1,keepdims=True)
    critical=factors[:,[3,4,5,9,17,19,20,21,22,23]]
    interactions=np.column_stack([
        factors[:,17]*base[:,2], factors[:,9]*base[:,3],
        factors[:,3]*(1-base[:,0]), factors[:,4]*(1-base[:,0]),
        risk_sum[:,0]*base[:,2], risk_sum[:,0]*base[:,3],
        protect_sum[:,0]*base[:,0], protect_sum[:,0]*(base[:,2]+base[:,3]),
    ])
    return np.column_stack((log,factors,risk_sum,protect_sum,critical,interactions))


def _model(c,class_weight):
    return make_pipeline(StandardScaler(),LogisticRegression(
        C=c,class_weight=class_weight,max_iter=5000,random_state=config.SEED,
    ))


def _probability(model,features):
    raw=model.predict_proba(features); result=np.zeros((len(features),4),np.float64)
    result[:,model[-1].classes_.astype(int)]=raw; return result


def _select(features,base,labels,groups):
    candidates=[]; folds=StratifiedGroupKFold(4,shuffle=True,random_state=6262)
    for c in (.001,.003,.01,.03,.1):
        for class_weight in (None,"balanced"):
            calibrated=np.zeros_like(base,dtype=np.float64)
            for fit,valid in folds.split(np.zeros(len(labels)),labels,groups):
                model=_model(c,class_weight).fit(features[fit],labels[fit]); calibrated[valid]=_probability(model,features[valid])
            for weight in (0.,.02,.05,.08,.10,.15):
                probability=(1-weight)*base+weight*calibrated
                prediction=probability.argmax(1)
                score=float(f1_score(labels,prediction,average="weighted",zero_division=0))
                candidates.append({"c":c,"class_weight":class_weight,"weight":weight,"inner_risk_f1":score})
    return max(candidates,key=lambda x:(x["inner_risk_f1"],-x["weight"],-x["c"])),candidates


def _metric(truth,risk,phrase):
    risk_f1=float(f1_score(truth,risk,average="weighted",zero_division=0)); phrase_f1=float(np.mean(phrase))
    return {"risk_f1":risk_f1,"phrase_f1":phrase_f1,"task1":task1_score(risk_f1,phrase_f1)}


def main():
    OUTPUT.mkdir(parents=True,exist_ok=True); frame=load_train_data().reset_index(drop=True)
    labels=frame.risk_label.to_numpy(np.int64); groups=frame.anon_user_id.astype(str).to_numpy(); train_idx,valid_idx=_outer_split(frame)
    inner_records,membership,outer_raw=_load_records(); global_indices=np.asarray([int(x["global_index"]) for x in inner_records])
    if not np.array_equal(np.sort(global_indices),np.sort(train_idx)): raise RuntimeError("V62 OOF records do not match outer training partition")
    by_index={int(x["global_index"]):x for x in inner_records}; ordered=[by_index[int(i)] for i in train_idx]
    train_base=np.vstack([x["old_probability"] for x in ordered]).astype(np.float64)
    factor_oof,_=_accepted_factor_oof(); train_features=_features(train_base,factor_oof[train_idx])
    selected,grid=_select(train_features,train_base,labels[train_idx],groups[train_idx])
    mapper=_model(float(selected["c"]),selected["class_weight"]).fit(train_features,labels[train_idx])

    base_probability,records,starts,ends,parameters=_current_baseline(frame,train_idx,valid_idx)
    trajectory=np.load(config.OUTPUT_DIR/"task1_factor_trajectory_v58"/"strict_predictions.npz")["trajectory_probability"]
    current_probability=.95*base_probability+.05*trajectory
    valid_features=_features(current_probability,factor_oof[valid_idx]); calibrated=_probability(mapper,valid_features)
    weight=float(selected["weight"]); candidate_probability=(1-weight)*current_probability+weight*calibrated
    texts=frame.text.iloc[valid_idx].astype(str).tolist(); truth=labels[valid_idx]
    baseline_risk=_risk(texts,current_probability); candidate_risk=_risk(texts,candidate_probability)
    baseline_evidence=_decode(records,baseline_risk,starts,ends,parameters); candidate_evidence=_decode(records,candidate_risk,starts,ends,parameters)
    gold=[list(frame.iloc[int(i)].evidence) for i in valid_idx]
    baseline_phrase=np.asarray([_post_phrase_f1(a,b) for a,b in zip(baseline_evidence,gold)])
    candidate_phrase=np.asarray([_post_phrase_f1(a,b) for a,b in zip(candidate_evidence,gold)])
    baseline=_metric(truth,baseline_risk,baseline_phrase); candidate=_metric(truth,candidate_risk,candidate_phrase)
    unique=np.unique(groups[valid_idx]); rng=np.random.default_rng(626262); deltas=[]
    for _ in range(4000):
        sampled=rng.choice(unique,len(unique),replace=True); pos=np.concatenate([np.flatnonzero(groups[valid_idx]==u) for u in sampled])
        old=f1_score(truth[pos],baseline_risk[pos],average="weighted",zero_division=0); new=f1_score(truth[pos],candidate_risk[pos],average="weighted",zero_division=0)
        deltas.append(task1_score(new,candidate_phrase[pos].mean())-task1_score(old,baseline_phrase[pos].mean()))
    deltas=np.asarray(deltas); bootstrap={"mean_delta":float(deltas.mean()),"p05_delta":float(np.quantile(deltas,.05)),
        "p95_delta":float(np.quantile(deltas,.95)),"positive_fraction":float((deltas>0).mean())}
    adopted=bool(candidate["task1"]>=baseline["task1"]+.003 and bootstrap["positive_fraction"]>=.80 and bootstrap["p05_delta"]>=0)
    payload={"training_version":TRAINING_VERSION,"evaluation_scope":"outer user-disjoint fold0; mapper fit on 1305 OOF posts",
        "method":"current-post continuous factor probabilities + baseline risk logits + clinically motivated interactions",
        "inner_selected":selected,"inner_grid_top10":sorted(grid,key=lambda x:x["inner_risk_f1"],reverse=True)[:10],
        "baseline_v58":baseline,"candidate":{**candidate,"changed_risk":int(np.sum(candidate_risk!=baseline_risk)),
            "confusion":confusion_matrix(truth,candidate_risk,labels=np.arange(4)).tolist()},"user_cluster_bootstrap":bootstrap,"adopted":adopted}
    np.savez_compressed(PREDICTIONS,valid_idx=valid_idx,baseline_probability=current_probability,candidate_probability=candidate_probability,
                        baseline_risk=baseline_risk,candidate_risk=candidate_risk)
    RESULTS.write_text(json.dumps(payload,indent=2),encoding="utf-8"); print(json.dumps(payload,indent=2),flush=True); return payload

if __name__=="__main__": main()
