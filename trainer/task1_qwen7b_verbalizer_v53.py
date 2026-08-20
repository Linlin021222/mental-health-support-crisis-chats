"""Low-memory Qwen2.5-7B boundary expert evaluated on untouched users (V53).

This experiment deliberately keeps the accepted V52 rationale-augmented expert
in the baseline.  The 7B verbalizer is therefore accepted only when it adds
value to the current production risk ensemble, rather than to an older model.
"""
from __future__ import annotations

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
from trainer import task1_qwen_verbalizer_v49 as qwen
from trainer.task1_atomic_v25 import _load_records
from trainer.task1_risk_only_v27 import _v18_evidence
from inference.task1_evidence_v4 import correct_risk_only
from utils.seed import seed_everything
from utils.task1_metric import task1_score


OUTPUT = config.OUTPUT_DIR / "task1_qwen7b_verbalizer_v53"
RESULTS = OUTPUT / "results.json"
ADAPTER = OUTPUT / "strict_adapter"
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
QWEN_WEIGHT = 0.10
V52_WEIGHT = 0.10
MAX_LENGTH = 128
COMPACT_PREFIX = (
    "Suicide risk label: A=no explicit suicide mention; "
    "B=suicidal thought or wish, no plan; C=plan, method, means, or self-harm; "
    "D=past or recent suicide attempt.\nPost:\n"
)
COMPACT_SUFFIX = "\nLabel:"


def _configure_qwen():
    qwen.MODEL_NAME = MODEL_NAME
    qwen.MAX_LENGTH = MAX_LENGTH
    qwen.BATCH_SIZE = 1
    qwen.ACCUMULATION = 16
    qwen.EPOCHS = 1
    qwen.USE_GRADIENT_CHECKPOINTING = True
    qwen.PREFIX = COMPACT_PREFIX
    qwen.SUFFIX = COMPACT_SUFFIX


def _softmax(values: np.ndarray, temperature: float) -> np.ndarray:
    logits = np.asarray(values, dtype=np.float64) / max(float(temperature), 1e-5)
    logits -= logits.max(1, keepdims=True)
    result = np.exp(logits)
    return result / result.sum(1, keepdims=True)


def _ensemble_prediction(texts, transformer, v52, qwen_probability,
                         lexical_decision, parameters, qwen_weight):
    """Apply the existing production calibration after a three-model blend."""
    neural = ((1.0 - V52_WEIGHT - qwen_weight) * transformer
              + V52_WEIGHT * v52 + qwen_weight * qwen_probability)
    lexical = _softmax(lexical_decision, float(parameters["temperature"]))
    probability = ((1.0 - float(parameters["lexical_weight"])) * neural
                   + float(parameters["lexical_weight"]) * lexical)
    logits = np.log(np.clip(probability, 1e-8, 1.0))
    logits[:, 0] += float(parameters.get("indicator_bias", 0.0))
    logits[:, 2] += float(parameters.get("behavior_bias", 0.0))
    logits[:, 3] += float(parameters.get("attempt_bias", 0.0))
    prediction = logits.argmax(1)
    return np.asarray([correct_risk_only(text, int(label))
                       for text, label in zip(texts, prediction)], dtype=np.int64)


def _metric(truth, risk, phrase):
    risk_f1 = float(f1_score(truth, risk, average="weighted", zero_division=0))
    phrase_f1 = float(np.mean(phrase))
    return {"risk_f1": risk_f1, "phrase_f1": phrase_f1,
            "task1": task1_score(risk_f1, phrase_f1)}


def memory_gate():
    """Run one genuine QLoRA backward pass before committing to fold training."""
    _configure_qwen()
    frame = load_train_data().reset_index(drop=True)
    labels = frame.risk_label.to_numpy(dtype=np.int64)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model, tokenizer, label_token_ids = qwen._model_and_tokenizer()
    data = qwen.PromptDataset(
        frame.text.iloc[:1].astype(str).tolist(), labels[:1], tokenizer,
    )
    batch = data[0]
    model.train()
    with torch.autocast(device_type="cuda", enabled=True):
        logits = qwen._verbalizer_logits(
            model, batch["input_ids"].unsqueeze(0).cuda(),
            batch["attention_mask"].unsqueeze(0).cuda(), label_token_ids,
        )
        loss = torch.nn.functional.cross_entropy(
            logits, batch["risk_labels"].view(1).cuda(),
        )
    loss.backward()
    peak = float(torch.cuda.max_memory_allocated() / 1024 ** 3)
    print(json.dumps({"memory_gate": "passed", "loss": float(loss.detach()),
                      "peak_allocated_gb": peak}, indent=2), flush=True)
    del model, data, batch, logits, loss
    torch.cuda.empty_cache()
    return peak


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("V53 requires CUDA")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    seed_everything(config.SEED + 5353)

    # Configure the reusable, tested V49 QLoRA implementation for an 8 GB GPU.
    _configure_qwen()

    frame = load_train_data().reset_index(drop=True)
    labels = frame.risk_label.to_numpy(dtype=np.int64)
    groups = frame.anon_user_id.astype(str).to_numpy()
    train_idx, valid_idx = next(StratifiedGroupKFold(
        n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED,
    ).split(np.zeros(len(frame)), labels, groups))

    torch.cuda.reset_peak_memory_stats()
    model, tokenizer, label_token_ids = qwen._model_and_tokenizer()
    prompt_data = qwen.PromptDataset(frame.text.astype(str).tolist(), labels, tokenizer)
    history = qwen._train(model, prompt_data, train_idx, labels, label_token_ids)
    qwen_probability = qwen._infer(
        model, prompt_data, valid_idx, label_token_ids, "V53 untouched outer users",
    )
    peak_memory_gb = float(torch.cuda.max_memory_allocated() / 1024 ** 3)
    print(f"V53 peak allocated CUDA memory={peak_memory_gb:.3f} GiB", flush=True)
    model.save_pretrained(ADAPTER)
    tokenizer.save_pretrained(ADAPTER)

    dataset = SuicideRiskDataset(config.CACHE_DIR / "train_cache.pt")
    _, _, outer_raw = _load_records()
    records = outer_raw["records"]
    transformer = _transformer_probability(dataset, valid_idx, records)
    v52_saved = torch.load(
        config.OUTPUT_DIR / "task1_rationale_augment_v52" / "strict_predictions.pt",
        map_location="cpu", weights_only=False,
    )
    if not np.array_equal(np.asarray(v52_saved["valid_idx"]), np.asarray(valid_idx)):
        raise RuntimeError("V52 and V53 outer user folds do not match")
    v52_probability = np.vstack([row["probability"] for row in v52_saved["rows"]])

    v36 = json.loads((config.OUTPUT_DIR / "task1_oof_risk_v36" / "calibration.json")
                     .read_text(encoding="utf-8"))
    lexical = _lexical_experts(frame, train_idx, valid_idx)[v36["expert"]]
    texts = frame.text.iloc[valid_idx].astype(str).tolist()
    baseline = _ensemble_prediction(
        texts, transformer, v52_probability, qwen_probability, lexical, v36, 0.0,
    )
    candidate = _ensemble_prediction(
        texts, transformer, v52_probability, qwen_probability, lexical, v36, QWEN_WEIGHT,
    )

    v18 = json.loads((config.OUTPUT_DIR / "task1_candidate_v18" / "results.json")
                     .read_text(encoding="utf-8"))
    v35 = json.loads((config.OUTPUT_DIR / "task1_oof_decoder_v35" / "calibration.json")
                     .read_text(encoding="utf-8"))
    evidence_parameters = (v35["parameters_by_predicted_risk"]
                           if v35.get("adopted", False)
                           else v18["evidence_parameters_by_predicted_risk"])
    seed2 = torch.load(
        config.OUTPUT_DIR / "task1_seed_ensemble_v14" / "seed2_valid.pt",
        map_location="cpu", weights_only=False,
    )["rows"]
    old_evidence = _v18_evidence(records, seed2, baseline, evidence_parameters)
    new_evidence = _v18_evidence(records, seed2, candidate, evidence_parameters)
    gold = [list(frame.iloc[int(i)].evidence) for i in valid_idx]
    old_phrase = np.asarray([_post_phrase_f1(x, y) for x, y in zip(old_evidence, gold)])
    new_phrase = np.asarray([_post_phrase_f1(x, y) for x, y in zip(new_evidence, gold)])
    truth = labels[valid_idx]
    base = _metric(truth, baseline, old_phrase)
    fixed = _metric(truth, candidate, new_phrase)

    unique_users = np.unique(groups[valid_idx])
    rng = np.random.default_rng(config.SEED + 5353)
    deltas = []
    for _ in range(4000):
        sampled = rng.choice(unique_users, size=len(unique_users), replace=True)
        positions = np.concatenate([
            np.flatnonzero(groups[valid_idx] == user) for user in sampled
        ])
        old_risk = f1_score(truth[positions], baseline[positions],
                            average="weighted", zero_division=0)
        new_risk = f1_score(truth[positions], candidate[positions],
                            average="weighted", zero_division=0)
        deltas.append(task1_score(new_risk, float(new_phrase[positions].mean()))
                      - task1_score(old_risk, float(old_phrase[positions].mean())))
    deltas = np.asarray(deltas)
    bootstrap = {"mean_delta": float(deltas.mean()),
                 "p05_delta": float(np.quantile(deltas, .05)),
                 "p95_delta": float(np.quantile(deltas, .95)),
                 "positive_fraction": float((deltas > 0).mean())}
    promising = bool(fixed["task1"] >= base["task1"] + .003
                     and bootstrap["positive_fraction"] >= .80)
    payload = {
        "training_version": "task1-qwen25-7b-verbalizer-v53",
        "model": MODEL_NAME,
        "evaluation_scope": "one untouched outer user fold; current V52 baseline",
        "method": {"quantization": "NF4 4-bit double quantization",
                   "max_length": qwen.MAX_LENGTH, "batch_size": qwen.BATCH_SIZE,
                   "gradient_checkpointing": True, "qwen_weight": QWEN_WEIGHT,
                   "v52_weight": V52_WEIGHT, "peak_memory_gb": peak_memory_gb},
        "history": history,
        "standalone": {"risk_f1": float(f1_score(
            truth, qwen_probability.argmax(1), average="weighted", zero_division=0))},
        "baseline_v52": base,
        "fixed_candidate": {**fixed,
                            "changed_predictions": int(np.sum(baseline != candidate)),
                            "confusion": confusion_matrix(
                                truth, candidate, labels=np.arange(4)).tolist()},
        "user_cluster_bootstrap": bootstrap,
        "promising_for_full_oof": promising,
        "adopted": False,
    }
    RESULTS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    return payload


if __name__ == "__main__":
    main()
