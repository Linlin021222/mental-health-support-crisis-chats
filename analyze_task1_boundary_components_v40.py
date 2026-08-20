"""Component ablation of V39 nominal and ordinal OOF probabilities."""
from __future__ import annotations

import json
import numpy as np
import torch
from sklearn.metrics import f1_score
from tqdm import tqdm

from analyze_task1_oof_risk_v36 import CACHE as V36_CACHE, _evidence_matrix, _predict
from configs.config import config
from datasets.dataset import SuicideRiskDataset
from inference.task1_evidence_v4 import correct_risk_only
from preprocess.preprocess import load_train_data
from trainer.task1_atomic_v25 import _load_records
from trainer.task1_boundary_model_v39 import BoundaryRiskModel, OUTPUT as V39_OUTPUT, _loader
from utils.task1_metric import task1_score


OUTPUT = config.OUTPUT_DIR / "task1_boundary_components_v40"
RESULTS = OUTPUT / "results.json"
TRAINING_VERSION = "task1-boundary-component-ablation-v40"


def _ordinal_probability(logits):
    cumulative = torch.cummin(torch.sigmoid(logits.float()), dim=1).values
    probability = torch.stack((
        1. - cumulative[:, 0], cumulative[:, 0] - cumulative[:, 1],
        cumulative[:, 1] - cumulative[:, 2], cumulative[:, 2],
    ), 1).clamp_min(1e-7)
    return (probability / probability.sum(1, keepdim=True)).cpu().numpy()


def _components(dataset, records):
    global_indices = np.asarray([int(row["global_index"]) for row in records])
    position = {int(index): pos for pos, index in enumerate(global_indices)}
    nominal = np.zeros((len(records), 4), dtype=np.float32)
    ordinal = np.zeros_like(nominal)
    device = torch.device("cuda")
    for fold in range(4):
        raw = torch.load(config.OUTPUT_DIR / "task1_oof_stack_v20" /
                         f"inner_fold{fold}_raw.pt", map_location="cpu",
                         weights_only=False)
        indices = np.asarray(raw["oof_idx"], dtype=np.int64)
        model = BoundaryRiskModel()
        saved = torch.load(V39_OUTPUT / f"fold{fold}_model.pt", map_location="cpu",
                           weights_only=True)
        model.load_state_dict(saved["model"]); model.to(device).eval()
        nominal_rows, ordinal_rows = [], []
        with torch.no_grad():
            for batch in tqdm(_loader(dataset, indices, False),
                              desc=f"V40 fold {fold} component inference"):
                with torch.autocast(device_type="cuda", enabled=True):
                    nom, ordered, _ = model(
                        batch["input_ids"].to(device), batch["attention_mask"].to(device)
                    )
                nominal_rows.append(torch.softmax(nom.float(), -1).cpu().numpy())
                ordinal_rows.append(_ordinal_probability(ordered))
        positions = [position[int(index)] for index in indices]
        nominal[positions] = np.vstack(nominal_rows)
        ordinal[positions] = np.vstack(ordinal_rows)
        del model; torch.cuda.empty_cache()
    return nominal, ordinal


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("V40 component ablation requires CUDA")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame = load_train_data().reset_index(drop=True)
    records, membership_map, _ = _load_records()
    global_indices = np.asarray([int(row["global_index"]) for row in records])
    truth = frame.risk_label.to_numpy()[global_indices]
    membership = np.asarray([membership_map[int(index)] for index in global_indices])
    dataset = SuicideRiskDataset(config.CACHE_DIR / "train_cache.pt")
    nominal, ordinal = _components(dataset, records)
    old = np.vstack([row["old_probability"] for row in records])
    evidence = _evidence_matrix(records)
    corrections = np.asarray([[correct_risk_only(row["text"], risk) for risk in range(4)]
                              for row in records], dtype=np.int64)
    saved = np.load(V36_CACHE, allow_pickle=True)
    names = saved["names"].tolist(); decisions = saved["decisions"]
    parameters = json.loads((config.OUTPUT_DIR / "task1_oof_risk_v36" / "calibration.json")
                            .read_text(encoding="utf-8"))
    lexical = decisions[names.index(parameters["expert"])]

    def metric(prediction, indices):
        indices = np.asarray(indices)
        risk = f1_score(truth[indices], prediction[indices], average="weighted",
                        zero_division=0)
        phrase = evidence[np.arange(len(prediction)), prediction][indices].mean()
        return float(risk), float(phrase), task1_score(risk, phrase)

    baseline_prediction = _predict(old, lexical, parameters, corrections)
    baseline = metric(baseline_prediction, np.arange(len(truth)))
    variants = []
    for ordinal_mix in (0., .10, .25, .50, 1.):
        new = (1. - ordinal_mix) * nominal + ordinal_mix * ordinal
        for new_weight in (.10, .25, .50, .75, 1.):
            probability = (1. - new_weight) * old + new_weight * new
            prediction = _predict(probability, lexical, parameters, corrections)
            full = metric(prediction, np.arange(len(truth)))
            folds = [metric(prediction, np.flatnonzero(membership == fold))[2]
                     for fold in range(4)]
            variants.append({"ordinal_mix": ordinal_mix, "new_model_weight": new_weight,
                             "risk_f1": full[0], "phrase_f1": full[1], "task1": full[2],
                             "fold_task1": folds})
    variants.sort(key=lambda row: row["task1"], reverse=True)
    payload = {"training_version": TRAINING_VERSION,
               "baseline_v36": {"risk_f1": baseline[0], "phrase_f1": baseline[1],
                                "task1": baseline[2]},
               "best_components": variants[:10],
               "improved": bool(variants[0]["task1"] > baseline[2])}
    RESULTS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    return payload


if __name__ == "__main__":
    main()
