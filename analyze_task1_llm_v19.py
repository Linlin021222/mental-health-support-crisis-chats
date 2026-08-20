"""Local constrained-LLM audit for Task 1 risk and evidence selection.

The LLM never writes submission evidence.  It can only select IDs from a
verbatim high-recall candidate pool, following the constrained highlighting
approach used by strong CLPsych 2024 systems.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from transformers import AutoModelForCausalLM, AutoTokenizer

from analyze_task1_risk_v10 import _probabilities
from baseline import _post_phrase_f1
from configs.config import config
from datasets.dataset import SuicideRiskDataset
from inference.task1_evidence_v4 import (
    apply_evidence_policy, correct_risk_only, decode_model_evidence,
    load_evidence_calibration,
)
from preprocess.preprocess import load_train_data
from trainer.task1_evidence_reranker_v13 import _candidate_pool
from utils.task1_metric import task1_score


MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
OUTPUT = config.OUTPUT_DIR / "task1_llm_v19"
RAW = OUTPUT / "strict_outputs.jsonl"
RESULTS = OUTPUT / "results.json"
CALIBRATION = OUTPUT / "calibration.json"
PROMPT_VERSION = "qwen25-constrained-rag-v19.2"
RISK_BOOST = 0.25  # fixed before seeing LLM holdout predictions
MAX_CANDIDATES = 16


def _short(value, maximum=360):
    value = " ".join(str(value).split())
    return value if len(value) <= maximum else value[:maximum].rsplit(" ", 1)[0] + "..."


def _retrieval_examples(frame, train_idx, valid_idx):
    """Retrieve one training-user example per label for each held-out post."""
    vectorizer = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), min_df=2,
        max_features=60000, sublinear_tf=True, dtype=np.float32,
    )
    train_text = frame.text.iloc[train_idx].astype(str).tolist()
    valid_text = frame.text.iloc[valid_idx].astype(str).tolist()
    train_matrix = vectorizer.fit_transform(train_text)
    valid_matrix = vectorizer.transform(valid_text)
    train_labels = frame.risk_label.to_numpy()[train_idx]
    result = []
    for row in valid_matrix:
        similarity = (row @ train_matrix.T).toarray().ravel()
        examples = []
        for label in range(config.NUM_RISK_CLASSES):
            local = np.flatnonzero(train_labels == label)
            selected_local = int(local[np.argmax(similarity[local])])
            source = frame.iloc[int(train_idx[selected_local])]
            examples.append({
                "risk": config.ID2RISK[label],
                "post": _short(source.text),
                "evidence": list(source.evidence)[:3],
            })
        result.append(examples)
    return result


def _prompt(record, candidates, examples):
    example_text = "\n".join(
        f"- {row['risk']} | post: {row['post']} | evidence: "
        f"{'; '.join(row['evidence']) if row['evidence'] else '[none]'}"
        for row in examples
    )
    candidate_text = "\n".join(
        f"{index}: {item['phrase']}" for index, item in enumerate(candidates, 1)
    ) or "[no candidates]"
    preliminary = config.ID2RISK[int(record["risk"])]
    return f"""You are checking a suicide-risk annotation for one Reddit post.

Use the annotation patterns in the examples, then apply these boundaries:
- Indicator: no current explicit self-suicidal expression by the author. Discussion of another person, quotation, hypothetical discussion, or a recovered/supportive stance can remain Indicator even when suicide words occur.
- Ideation: the author expresses their own wish/thought of death or suicide, without a method, concrete plan, preparation, or attempt.
- Behavior: the author expresses a current/future method, plan, preparation, access to means, or suicidal/self-harm behavior, but does not report an actually completed suicide attempt.
- Attempt: the author explicitly reports an actual past or recent suicide attempt. Merely considering, planning, almost doing, or being stopped before acting is Behavior, not Attempt.

Training-user examples:
{example_text}

Preliminary classifier label: {preliminary}. Verify it; correct it only when the post clearly crosses a boundary.

POST:
{record['text']}

VERBATIM EVIDENCE CANDIDATES:
{candidate_text}

Allowed risk values are Indicator, Ideation, Behavior, and Attempt. Select exactly one.
Return exactly one compact JSON object. Example of the required syntax:
{{"risk":"Ideation","evidence_ids":[2]}}
The example value Ideation is not a suggested answer. Never output vertical bars or multiple risk names.
Choose at most 3 IDs. Choose only short candidates that directly establish the selected risk. For Indicator normally return an empty list. Never copy or rewrite text in the answer."""


def _parse(text, candidate_count, fallback_risk):
    risk = None; ids = []
    match = re.search(r"\{.*?\}", text, flags=re.DOTALL)
    if match:
        try:
            payload = json.loads(match.group(0))
            value = str(payload.get("risk", "")).strip().casefold()
            for label in config.RISK_LABELS:
                if value == label.casefold():
                    risk = config.RISK_LABELS[label]
                    break
            raw_ids = payload.get("evidence_ids", [])
            if isinstance(raw_ids, list):
                ids = [int(value) for value in raw_ids if str(value).strip().isdigit()]
        except (ValueError, TypeError, json.JSONDecodeError):
            pass
    if risk is None:
        # Only accept a free-form fallback when the answer contains one
        # unambiguous label.  Prompts/examples can otherwise make a malformed
        # answer mention all four labels and silently collapse it to Indicator.
        mentioned = {
            match.group(1).title()
            for match in re.finditer(
                r"\b(Indicator|Ideation|Behavior|Attempt)\b", text, re.I
            )
        }
        risk = (config.RISK_LABELS[next(iter(mentioned))]
                if len(mentioned) == 1 else int(fallback_risk))
    unique = []
    for value in ids:
        if 1 <= value <= candidate_count and value not in unique:
            unique.append(value)
        if len(unique) == 3:
            break
    return int(risk), unique


def _load_resume():
    rows = {}
    if not RAW.exists():
        return rows
    for line in RAW.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
            if row.get("prompt_version") == PROMPT_VERSION:
                rows[str(row["row_id"])] = row
        except (ValueError, KeyError):
            continue
    return rows


def _baseline_evidence(record, risk, calibration):
    spans = decode_model_evidence(
        record["text"], record["offsets"], record["start"], record["end"],
        threshold=float(calibration["threshold"]),
        max_tokens=int(calibration["max_tokens"]),
        end_policy=str(calibration["end_policy"]), limit=5,
    )
    return apply_evidence_policy(
        record["text"], int(risk), spans,
        policy=str(calibration["cue_policy"]), topk=int(calibration["topk"]),
    )


def _bootstrap(groups, truth, base_risk, base_phrase, new_risk, new_phrase, rounds=4000):
    rng = np.random.default_rng(config.SEED + 1919); users = np.unique(groups); values = []
    for _ in range(rounds):
        sampled_users = rng.choice(users, len(users), replace=True)
        indices = np.concatenate([np.flatnonzero(groups == user) for user in sampled_users])
        old_f1 = f1_score(truth[indices], base_risk[indices], average="weighted", zero_division=0)
        new_f1 = f1_score(truth[indices], new_risk[indices], average="weighted", zero_division=0)
        values.append(
            task1_score(new_f1, float(base_phrase[indices].mean() if new_phrase is None else new_phrase[indices].mean()))
            - task1_score(old_f1, float(base_phrase[indices].mean()))
        )
    values = np.asarray(values)
    return {
        "mean_delta": float(values.mean()),
        "p05_delta": float(np.quantile(values, 0.05)),
        "p95_delta": float(np.quantile(values, 0.95)),
        "positive_fraction": float((values > 0).mean()),
    }


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame = load_train_data().reset_index(drop=True)
    dataset = SuicideRiskDataset(config.CACHE_DIR / "train_cache.pt")
    labels = frame.risk_label.to_numpy(); groups = frame.anon_user_id.astype(str).to_numpy()
    train_idx, valid_idx = next(StratifiedGroupKFold(
        n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED,
    ).split(np.zeros(len(frame)), labels, groups))
    raw = torch.load(
        config.OUTPUT_DIR / "task1_evidence_v4" / "fold0_raw.pt",
        map_location="cpu", weights_only=False,
    )
    if not np.array_equal(np.asarray(raw["valid_idx"]), valid_idx):
        raise ValueError("V19 strict rows differ from V4")
    records = raw["records"]
    examples = _retrieval_examples(frame, train_idx, valid_idx)
    pools = [_candidate_pool(record, int(record["risk"]))[:MAX_CANDIDATES] for record in records]

    resumed = _load_resume(); missing = [
        index for index, record in enumerate(records) if str(record["row_id"]) not in resumed
    ]
    print(f"V19 constrained LLM: resumed={len(resumed)} missing={len(missing)}", flush=True)
    if missing:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        ).to("cuda").eval()
        for position, index in enumerate(missing, 1):
            record = records[index]; candidates = pools[index]
            messages = [
                {"role": "system", "content": "You are a precise clinical text annotation verifier."},
                {"role": "user", "content": _prompt(record, candidates, examples[index])},
            ]
            rendered = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            inputs = tokenizer(
                rendered, return_tensors="pt", truncation=True, max_length=6144,
            ).to("cuda")
            with torch.inference_mode():
                generated = model.generate(
                    **inputs, max_new_tokens=48, do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            answer = tokenizer.decode(
                generated[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True,
            ).strip()
            risk, selected = _parse(answer, len(candidates), int(record["risk"]))
            row = {
                "prompt_version": PROMPT_VERSION, "row_id": str(record["row_id"]),
                "risk": risk, "evidence_ids": selected, "answer": answer,
            }
            with RAW.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            resumed[row["row_id"]] = row
            if position % 10 == 0 or position == len(missing):
                print(f"V19 LLM inference {position}/{len(missing)}", flush=True)
        del model
        torch.cuda.empty_cache()

    stable_probability = _probabilities(dataset, train_idx, valid_idx, records)
    calibration = load_evidence_calibration()
    truth = labels[valid_idx]; local_groups = groups[valid_idx]
    baseline_risk = np.asarray([int(record["risk"]) for record in records])
    llm_risk = np.asarray([resumed[str(record["row_id"])]["risk"] for record in records])
    logits = np.log(np.asarray(stable_probability).clip(1e-8, 1.0))
    logits[np.arange(len(logits)), llm_risk] += RISK_BOOST
    fused_risk = np.asarray([
        correct_risk_only(record["text"], int(value))
        for record, value in zip(records, logits.argmax(1))
    ])

    baseline_scores = np.zeros(len(records), dtype=np.float32)
    llm_scores = np.zeros(len(records), dtype=np.float32)
    conservative_scores = np.zeros(len(records), dtype=np.float32)
    selected_counts = []
    for index, (record, candidates) in enumerate(zip(records, pools)):
        old = _baseline_evidence(record, baseline_risk[index], calibration)
        selected = resumed[str(record["row_id"])]["evidence_ids"]
        chosen = [candidates[value - 1]["phrase"] for value in selected]
        if llm_risk[index] == config.RISK_LABELS["Indicator"]:
            chosen = []
        baseline_scores[index] = _post_phrase_f1(old, record["gold"])
        llm_scores[index] = _post_phrase_f1(chosen, record["gold"])
        # Predeclared conservative policy: only use LLM spans when its risk
        # agrees with the fused label and it returned at least one valid ID.
        conservative = (chosen if chosen and llm_risk[index] == fused_risk[index] else
                        _baseline_evidence(record, fused_risk[index], calibration))
        conservative_scores[index] = _post_phrase_f1(conservative, record["gold"])
        selected_counts.append(len(chosen))

    def metric(risk, phrase):
        risk_f1 = float(f1_score(truth, risk, average="weighted", zero_division=0))
        phrase_f1 = float(np.mean(phrase))
        return {"risk_f1": risk_f1, "phrase_f1": phrase_f1,
                "task1": task1_score(risk_f1, phrase_f1)}

    baseline = metric(baseline_risk, baseline_scores)
    standalone = metric(llm_risk, llm_scores)
    candidate = metric(fused_risk, conservative_scores)
    bootstrap = _bootstrap(
        local_groups, truth, baseline_risk, baseline_scores,
        fused_risk, conservative_scores,
    )
    adopted = bool(
        candidate["task1"] >= baseline["task1"] + 0.005
        and bootstrap["positive_fraction"] >= 0.80
    )
    payload = {
        "training_version": PROMPT_VERSION,
        "model": MODEL_NAME,
        "evaluation_scope": "one user-disjoint holdout; no LLM parameter search",
        "predeclared_risk_boost": RISK_BOOST,
        "baseline": baseline,
        "llm_standalone": {**standalone,
            "confusion": confusion_matrix(truth, llm_risk, labels=np.arange(4)).tolist()},
        "conservative_fusion": {**candidate,
            "confusion": confusion_matrix(truth, fused_risk, labels=np.arange(4)).tolist(),
            "mean_selected_evidence": float(np.mean(selected_counts)),
            "evidence_improved_posts": int((conservative_scores > baseline_scores).sum()),
            "evidence_worsened_posts": int((conservative_scores < baseline_scores).sum())},
        "user_cluster_bootstrap": bootstrap,
        "adopted": adopted,
    }
    RESULTS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    CALIBRATION.write_text(json.dumps({
        "training_version": PROMPT_VERSION, "model": MODEL_NAME,
        "adopted": adopted, "risk_boost": RISK_BOOST,
        "strict_task1": candidate["task1"],
        "bootstrap_positive_fraction": bootstrap["positive_fraction"],
    }, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    return payload


if __name__ == "__main__":
    main()
