"""Strict user-disjoint gate for the Task-1 negation polarity correction."""
from __future__ import annotations

import json

import numpy as np
import torch
from sklearn.metrics import f1_score

from baseline import _post_phrase_f1
from configs.config import config
from inference.task1_evidence_v4 import (
    apply_evidence_policy, decode_model_evidence,
)
from inference.task1_polarity_v63 import CALIBRATION, polarity_candidate


OUTPUT = CALIBRATION.parent
RESULTS = OUTPUT / "results.json"
RAW = config.OUTPUT_DIR / "task1_evidence_v4" / "fold0_raw.pt"
V4 = config.OUTPUT_DIR / "task1_evidence_v4" / "calibration.json"
TRAINING_VERSION = "task1-negation-polarity-v63"


def _metric(truth, risk, phrase):
    risk_f1 = float(f1_score(truth, risk, average="weighted", zero_division=0))
    phrase_f1 = float(np.mean(phrase))
    return {
        "risk_f1": risk_f1, "phrase_f1": phrase_f1,
        "task1": float((.4 * risk_f1 + .3 * phrase_f1) / .7),
    }


def _bootstrap(records, base_risk, new_risk, base_phrase, new_phrase, draws=4000):
    users = np.asarray([str(row["user"]) for row in records])
    truth = np.asarray([int(row["truth"]) for row in records])
    unique = np.unique(users); rng = np.random.default_rng(636363)
    values = []
    for _ in range(draws):
        sampled = rng.choice(unique, len(unique), replace=True)
        idx = np.concatenate([np.flatnonzero(users == user) for user in sampled])
        old = _metric(truth[idx], base_risk[idx], base_phrase[idx])["task1"]
        new = _metric(truth[idx], new_risk[idx], new_phrase[idx])["task1"]
        values.append(new - old)
    values = np.asarray(values)
    return {
        "mean_delta": float(values.mean()),
        "p05_delta": float(np.quantile(values, .05)),
        "p95_delta": float(np.quantile(values, .95)),
        "positive_fraction": float((values > 0).mean()),
    }


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    records = torch.load(RAW, map_location="cpu", weights_only=False)["records"]
    decoder = json.loads(V4.read_text(encoding="utf-8"))
    truth = np.asarray([int(row["truth"]) for row in records])
    base_risk = np.asarray([int(row["risk"]) for row in records])
    new_risk = base_risk.copy(); base_phrase = []; new_phrase = []; changed = []
    for index, row in enumerate(records):
        spans = decode_model_evidence(
            row["text"], row["offsets"], row["start"], row["end"],
            threshold=float(decoder["threshold"]),
            max_tokens=int(decoder["max_tokens"]),
            end_policy=str(decoder["end_policy"]), limit=5,
        )
        old = apply_evidence_policy(
            row["text"], int(base_risk[index]), spans,
            policy=str(decoder["cue_policy"]), topk=int(decoder["topk"]),
        )
        candidate = polarity_candidate(row["text"], int(base_risk[index]))
        if candidate is None:
            new = old
        else:
            new_risk[index], new = candidate
            changed.append({
                "row_id": row["row_id"], "truth": int(row["truth"]),
                "old_risk": int(base_risk[index]), "new_risk": int(new_risk[index]),
                "evidence": new,
            })
        base_phrase.append(_post_phrase_f1(old, row["gold"]))
        new_phrase.append(_post_phrase_f1(new, row["gold"]))
    base_phrase = np.asarray(base_phrase); new_phrase = np.asarray(new_phrase)
    baseline = _metric(truth, base_risk, base_phrase)
    candidate = _metric(truth, new_risk, new_phrase)
    bootstrap = _bootstrap(
        records, base_risk, new_risk, base_phrase, new_phrase,
    )
    adopted = bool(
        candidate["task1"] >= baseline["task1"] + .001
        and bootstrap["positive_fraction"] >= .75
        and bootstrap["p05_delta"] >= 0
        and all(row["truth"] == row["new_risk"] for row in changed)
    )
    payload = {
        "training_version": TRAINING_VERSION,
        "evaluation": "strict user-disjoint outer fold; fixed pre-registered polarity rule",
        "baseline": baseline, "candidate": candidate,
        "delta": candidate["task1"] - baseline["task1"],
        "changed": changed, "user_cluster_bootstrap": bootstrap,
        "adopted": adopted,
    }
    RESULTS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    CALIBRATION.write_text(json.dumps({
        "training_version": TRAINING_VERSION, "adopted": adopted,
        "strict_baseline_task1": baseline["task1"],
        "strict_candidate_task1": candidate["task1"],
        "changed_posts": len(changed), "bootstrap": bootstrap,
    }, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    return payload


if __name__ == "__main__":
    main()
