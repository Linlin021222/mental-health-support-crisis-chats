"""Diversity-controlled local Qwen7B counterfactual generation (V57)."""
from __future__ import annotations

import json

import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold
from tqdm import tqdm

from configs.config import config
from preprocess.preprocess import load_train_data
from trainer.task1_local_counterfactual_v56 import (
    ANCHORS, TRANSFORMS, _json_object, _load_model, _prompt, _validate,
)
from utils.seed import seed_everything


OUTPUT = config.OUTPUT_DIR / "task1_local_diverse_cf_v57"
SYNTHETIC_FILE = OUTPUT / "synthetic.json"
V56_FILE = config.OUTPUT_DIR / "task1_local_counterfactual_v56" / "synthetic.json"
TRAINING_VERSION = "task1-local-diverse-counterfactual-v57"
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
SEED = 575757


@torch.inference_mode()
def generate(target_per_transform=30, max_attempts=2, force=False):
    if not torch.cuda.is_available():
        raise RuntimeError("V57 local generation requires CUDA")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if SYNTHETIC_FILE.exists() and not force:
        payload = json.loads(SYNTHETIC_FILE.read_text(encoding="utf-8"))
        print(f"V57 generation resumed: {len(payload['accepted'])} accepted", flush=True)
        return payload
    seed_everything(SEED)
    frame = load_train_data().reset_index(drop=True)
    labels = frame.risk_label.to_numpy(dtype=np.int64)
    groups = frame.anon_user_id.astype(str).to_numpy()
    train_idx, _ = next(StratifiedGroupKFold(
        n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED,
    ).split(np.zeros(len(frame)), labels, groups))
    # V57 measures context diversity, so none of the V56 source posts may be
    # selected again, whether their earlier generation passed or failed.
    excluded = set()
    if V56_FILE.exists():
        old = json.loads(V56_FILE.read_text(encoding="utf-8"))
        excluded = {int(row["source_index"])
                    for key in ("accepted", "rejected") for row in old[key]}
    rng = np.random.default_rng(SEED)
    jobs = []
    for transform, spec in TRANSFORMS.items():
        candidates = np.asarray([
            int(index) for index in train_idx
            if int(index) not in excluded
            and labels[int(index)] == spec["source"]
            and bool(frame.iloc[int(index)].evidence)
            and 20 <= len(str(frame.iloc[int(index)].text).split()) <= 220
        ])
        rng.shuffle(candidates)
        jobs.extend((transform, int(index))
                    for index in candidates[:target_per_transform])
    model, tokenizer = _load_model()
    accepted, rejected = [], []
    torch.cuda.reset_peak_memory_stats()
    for job_number, (transform, index) in enumerate(
            tqdm(jobs, desc="V57 diverse local generation")):
        row = frame.iloc[index]; target = int(TRANSFORMS[transform]["target"])
        attempts = []
        clean = None
        for attempt in range(max_attempts):
            anchor = ANCHORS[target][(job_number + attempt + 1) % len(ANCHORS[target])]
            prompt = _prompt(row.text, row.evidence, transform, anchor)
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False,
                add_generation_prompt=True,
            )
            encoded = tokenizer(
                rendered, return_tensors="pt", truncation=True, max_length=320,
            ).to("cuda")
            torch.manual_seed(SEED + job_number * max_attempts + attempt)
            output = model.generate(
                **encoded, max_new_tokens=170, do_sample=True, temperature=.62,
                top_p=.90, repetition_penalty=1.05,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            raw = tokenizer.decode(
                output[0, encoded.input_ids.shape[1]:], skip_special_tokens=True,
            )
            try:
                parsed = _json_object(raw)
                clean, reason = _validate(parsed, row.text, target, anchor)
            except Exception as error:
                clean, reason = None, f"parse:{type(error).__name__}:{error}"
            attempts.append({"attempt": attempt + 1, "reason": reason, "raw": raw})
            if clean is not None:
                break
        audit = {
            "transform": transform, "source_index": index,
            "source_row_id": str(row.row_id), "source_risk": int(labels[index]),
            "target_risk": target, "attempt_count": len(attempts),
        }
        if clean is None:
            rejected.append({**audit, "attempts": attempts})
        else:
            factors = np.asarray(row.factor_vector, dtype=np.float32).copy()
            factors[9] = 1.; factors[17] = 1. if target == 2 else 0.
            accepted.append({
                **audit, **clean, "factor_vector": factors.astype(float).tolist(),
                "audit_attempts": attempts[:-1],
            })
        print(f"V57 {job_number + 1}/{len(jobs)} accepted={len(accepted)} "
              f"rejected={len(rejected)}", flush=True)
    payload = {
        "training_version": TRAINING_VERSION, "model": MODEL_NAME,
        "scope": "outer training users only; V56 sources excluded",
        "requested_per_transform": int(target_per_transform),
        "max_attempts": int(max_attempts), "excluded_v56_sources": len(excluded),
        "accepted": accepted, "rejected": rejected,
        "acceptance_rate": float(len(accepted) / max(1, len(jobs))),
        "peak_memory_gb": float(torch.cuda.max_memory_allocated() / 1024 ** 3),
    }
    SYNTHETIC_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items()
                      if key not in ("accepted", "rejected")}, indent=2), flush=True)
    del model; torch.cuda.empty_cache()
    return payload


if __name__ == "__main__":
    generate()
