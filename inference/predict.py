"""Generate a competition-format submission with risk, evidence, and factors."""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from configs.config import config
from datasets.dataset import SuicideRiskDataset
from datasets.collator import SuicideRiskCollator
from datasets.cache_builder import build_cache
from models.multitask_model import SuicideRiskMultiTaskModel
from baseline import _apply_task1_rules
from inference.task1_evidence_v4 import (
    apply_evidence_policy, correct_risk_only, decode_model_evidence,
    load_evidence_calibration,
)
from utils.factor_calibration import (
    load_factor_calibration, apply_calibrated_thresholds, blend_cpu_factor_probabilities,
    apply_prior_topk, cpu_factor_probabilities,
)


def decode_evidence(text, offsets, start_logits, end_logits):
    start = torch.sigmoid(start_logits).cpu().numpy()
    end = torch.sigmoid(end_logits).cpu().numpy()
    candidates = []
    for chunk, chunk_offsets in enumerate(offsets):
        starts = np.where(start[chunk] >= config.EVIDENCE_THRESHOLD)[0]
        ends = np.where(end[chunk] >= config.EVIDENCE_THRESHOLD)[0]
        for s in starts:
            valid = [e for e in ends
                     if s <= e <= s + config.MAX_EVIDENCE_TOKENS
                     and chunk_offsets[s][1] > chunk_offsets[s][0]]
            if valid:
                e = min(valid, key=lambda x: x - s)
                a, b = chunk_offsets[s][0], chunk_offsets[e][1]
                phrase = text[a:b].strip()
                score = float(start[chunk, s] * end[chunk, e])
                if phrase:
                    candidates.append((score, phrase))
    selected = []
    for _, phrase in sorted(candidates, reverse=True):
        norm = " ".join(phrase.lower().split())
        if not any(norm in p.lower() or p.lower() in norm for p in selected):
            selected.append(phrase)
        if len(selected) == config.MAX_EVIDENCE_PHRASES:
            break
    return selected


@torch.no_grad()
def predict(checkpoint=None, task2_experiment=None, task1_experiment=None):
    cache_file = config.CACHE_DIR / "test_cache.pt"
    # Rebuild from leaderboard.xlsx for a final submission so an old test
    # cache (or a cache made with different chunk settings) cannot silently be
    # used for prediction.
    build_cache(train=False)
    # Submission prediction should use the checkpoint trained on all labelled
    # posts.  A caller can still explicitly pass a strict-fold checkpoint for
    # diagnostics.
    checkpoint = Path(checkpoint or (config.OUTPUT_DIR / "full_train_model.pt"))
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}. Run training first.")
    device = torch.device(config.DEVICE)
    model = SuicideRiskMultiTaskModel().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    loader = DataLoader(SuicideRiskDataset(cache_file), batch_size=config.BATCH_SIZE,
                        shuffle=False, collate_fn=SuicideRiskCollator(), num_workers=config.NUM_WORKERS)
    evidence_v4 = load_evidence_calibration()
    if evidence_v4 is not None:
        print(
            "Task 1 evidence-v4: "
            f"threshold={evidence_v4['threshold']:.2f}, "
            f"max_tokens={evidence_v4['max_tokens']}, "
            f"end={evidence_v4['end_policy']}, "
            f"cues={evidence_v4['cue_policy']}, topk={evidence_v4['topk']}"
        )
    raw_rows = []
    all_factor_probs = []
    for batch in loader:
        outputs = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        risk_probabilities = torch.softmax(outputs["risk_logits"], dim=-1).cpu().numpy()
        factor_probs = torch.sigmoid(outputs["factor_logits"]).cpu().numpy()
        for i in range(len(batch["row_id"])):
            if evidence_v4 is None:
                evidence = decode_evidence(
                    batch["texts"][i], batch["offset_mappings"][i],
                    outputs["start_logits"][i], outputs["end_logits"][i]
                )
            else:
                evidence = decode_model_evidence(
                    batch["texts"][i], batch["offset_mappings"][i],
                    outputs["start_logits"][i], outputs["end_logits"][i],
                    threshold=float(evidence_v4["threshold"]),
                    max_tokens=int(evidence_v4["max_tokens"]),
                    end_policy=str(evidence_v4["end_policy"]), limit=5,
                )
            all_factor_probs.append(factor_probs[i])
            raw_rows.append({"row_id": batch["row_id"][i], "_text": batch["texts"][i],
                             "_risk_probability": risk_probabilities[i],
                             "_evidence": evidence,
                             "_offsets": batch["offset_mappings"][i],
                             "_start_logits": outputs["start_logits"][i].float().cpu(),
                             "_end_logits": outputs["end_logits"][i].float().cpu()})
    # Release the Task 1 encoder before loading the independent Task 2 model;
    # both models fit on an 8 GB GPU sequentially but should not coexist there.
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    from inference.task1_v2_predictor import task1_v2_probabilities
    v2_row_ids, v2_probabilities = task1_v2_probabilities()
    if v2_probabilities is not None:
        v2_by_id = dict(zip(v2_row_ids, v2_probabilities))
        weight = float(config.TASK1_V2_ENSEMBLE_WEIGHT)
        print(f"Task 1 risk ensemble: legacy={1-weight:.2f}, ordinal-V2={weight:.2f}")
    else:
        v2_by_id = None
    from inference.task1_rationale_v52_predictor import task1_v52_probabilities
    v52_row_ids, v52_probabilities, v52_weight = task1_v52_probabilities()
    if v52_probabilities is not None:
        v52_by_id = dict(zip(v52_row_ids, v52_probabilities))
        print(f"Task 1 V52 rationale-augmented risk expert: weight={v52_weight:.2f}")
    else:
        v52_by_id = None
    from inference.task1_local_diverse_v57_predictor import task1_v57_outputs
    v57_rows, v57_risk_weight, v57_evidence_weight = task1_v57_outputs()
    if v57_rows is not None:
        v57_by_id = {row["row_id"]: row for row in v57_rows}
        print(
            "Task 1 V57 local diverse counterfactual expert: "
            f"risk_weight={v57_risk_weight:.2f}, "
            f"evidence_weight={v57_evidence_weight:.2f}"
        )
    else:
        v57_by_id = None
    from inference.task1_factor_trajectory_v58_predictor import task1_v58_probabilities
    v58_row_ids, v58_probabilities, v58_weight = task1_v58_probabilities()
    if v58_probabilities is not None:
        v58_by_id = dict(zip(map(str, v58_row_ids), v58_probabilities))
        print(
            "Task 1 V58 experimental paper factor-trajectory expert: "
            f"weight={v58_weight:.2f}"
        )
    else:
        v58_by_id = None
    from inference.task1_v18_predictor import (
        V36_LEXICAL_MODEL, load_v18_calibration, load_v36_risk_calibration,
        v18_lexical_probabilities, v18_seed2_evidence,
    )
    v18_calibration = load_v18_calibration()
    v36_risk_calibration = load_v36_risk_calibration()
    v35_parameters = None
    if v18_calibration is not None:
        v35_file = config.OUTPUT_DIR / "task1_oof_decoder_v35" / "calibration.json"
        if v35_file.exists():
            import json
            v35_calibration = json.loads(v35_file.read_text(encoding="utf-8"))
            if v35_calibration.get("adopted", False):
                v35_parameters = v35_calibration["parameters_by_predicted_risk"]
        risk_calibration = v36_risk_calibration or v18_calibration
        v18_lexical = v18_lexical_probabilities(
            [row["_text"] for row in raw_rows], risk_calibration,
            V36_LEXICAL_MODEL if v36_risk_calibration is not None else None,
        )
        v18_lexical_by_id = {
            str(row["row_id"]): probability
            for row, probability in zip(raw_rows, v18_lexical)
        }
        v18_seed2_by_id = {
            row["row_id"]: row for row in v18_seed2_evidence()
        }
        lexical_description = ("OOF-selected C=0.5 lexical risk"
                               if v36_risk_calibration is not None else
                               "C=0.25 lexical risk")
        print(
            f"Task 1 V18 active: full DeBERTa/ordinal + {lexical_description} "
            "+ independent-seed label-conditional evidence"
        )
        if v35_parameters is not None:
            print(
                "Task 1 V35 active: full nested-OOF label-conditional evidence "
                "calibration replaces the overfit V18 decoder parameters"
            )
        if v36_risk_calibration is not None:
            print(
                "Task 1 V36 active: full nested-OOF SVC risk calibration "
                "replaces the overfit V18 risk fusion parameters"
            )
    else:
        v18_lexical_by_id = v18_seed2_by_id = None
    from inference.task1_tfidf_hybrid import task1_tfidf_probabilities
    if v18_calibration is None:
        tfidf_probabilities, tfidf_calibration = task1_tfidf_probabilities(
            [row["_text"] for row in raw_rows]
        )
    else:
        tfidf_probabilities, tfidf_calibration = None, None
    if tfidf_probabilities is not None:
        tfidf_by_id = {
            str(row["row_id"]): probability
            for row, probability in zip(raw_rows, tfidf_probabilities)
        }
        tfidf_weight = float(tfidf_calibration["lexical_weight"])
        print(
            f"Task 1 TF-IDF risk hybrid: Transformer={1.0-tfidf_weight:.2f}, "
            f"TF-IDF SVC={tfidf_weight:.2f}"
        )
    else:
        tfidf_by_id = None
    from inference.task1_cv_predictor import task1_cv_predictions
    if v18_calibration is None:
        cv_row_ids, cv_probabilities, cv_evidence, cv_calibration = task1_cv_predictions()
    else:
        cv_row_ids, cv_probabilities, cv_evidence, cv_calibration = None, None, None, None
    if cv_probabilities is not None:
        cv_probability_by_id = {
            str(row_id): probability for row_id, probability in zip(cv_row_ids, cv_probabilities)
        }
        cv_evidence_by_id = {
            str(row_id): evidence for row_id, evidence in zip(cv_row_ids, cv_evidence)
        }
        cv_weight = float(cv_calibration.get("test_weight", config.TASK1_CV_TEST_WEIGHT))
        cv_use_risk = bool(cv_calibration.get("use_risk", True))
        cv_use_evidence = bool(cv_calibration.get("use_evidence", False))
        print(
            f"Task 1 five-fold ensemble: stable={1-cv_weight:.2f}, "
            f"CV={cv_weight:.2f}, CV risk={cv_use_risk}, CV evidence={cv_use_evidence}"
        )
    else:
        cv_probability_by_id = cv_evidence_by_id = None
    qwen_by_id = None
    qwen_confidence_threshold = None
    qwen_blend_weight = None
    qwen_class_bias = None
    if task1_experiment in (
        "qwen3-8b", "qwen3-8b-blend", "qwen3-8b-attempt-calibrated"
    ):
        from inference.task1_qwen3_8b_predictor import (
            ATTEMPT_CALIBRATED_BLEND_WEIGHT, ATTEMPT_CALIBRATED_CLASS_BIAS,
            ATTEMPT_CALIBRATED_TEMPERATURE, BLEND_WEIGHT,
            CONFIDENCE_THRESHOLD, TEMPERATURE,
            qwen3_8b_probabilities,
        )
        qwen_temperature = (
            ATTEMPT_CALIBRATED_TEMPERATURE
            if task1_experiment == "qwen3-8b-attempt-calibrated"
            else TEMPERATURE
        )
        qwen_row_ids, qwen_probabilities = qwen3_8b_probabilities(
            temperature=qwen_temperature
        )
        qwen_by_id = dict(zip(map(str, qwen_row_ids), qwen_probabilities))
        if task1_experiment == "qwen3-8b-attempt-calibrated":
            qwen_blend_weight = float(ATTEMPT_CALIBRATED_BLEND_WEIGHT)
            qwen_class_bias = np.asarray(
                ATTEMPT_CALIBRATED_CLASS_BIAS, dtype=np.float64
            )
            print(
                "Task 1 Qwen3-8B Attempt-calibrated blend: "
                f"current={1.0-qwen_blend_weight:.2f}, "
                f"Qwen={qwen_blend_weight:.2f}, temperature={qwen_temperature:.2f}, "
                f"class_bias={ATTEMPT_CALIBRATED_CLASS_BIAS}",
                flush=True,
            )
        elif task1_experiment == "qwen3-8b-blend":
            qwen_blend_weight = float(BLEND_WEIGHT)
            print(
                "Task 1 Qwen3-8B probability blend: "
                f"current={1.0-BLEND_WEIGHT:.2f}, Qwen={BLEND_WEIGHT:.2f}, "
                f"temperature={TEMPERATURE:.2f}",
                flush=True,
            )
        else:
            qwen_confidence_threshold = float(CONFIDENCE_THRESHOLD)
            print(
                "Task 1 Qwen3-8B high-confidence gate: "
                f"temperature={TEMPERATURE:.2f}, confidence>={CONFIDENCE_THRESHOLD:.2f}",
                flush=True,
            )
    from inference.task1_polarity_v63 import apply_polarity_correction
    polarity_changes = 0
    qwen_changes = 0
    for row in raw_rows:
        risk_probability = row.pop("_risk_probability")
        if v2_by_id is not None:
            risk_probability = (
                (1.0-weight)*risk_probability + weight*v2_by_id[row["row_id"]]
            )
        if v52_by_id is not None and v57_by_id is not None:
            risk_probability = (
                (1.0-v52_weight-v57_risk_weight)*risk_probability
                + v52_weight*v52_by_id[row["row_id"]]
                + v57_risk_weight*v57_by_id[str(row["row_id"])]["probability"]
            )
        elif v52_by_id is not None:
            risk_probability = (
                (1.0-v52_weight)*risk_probability
                + v52_weight*v52_by_id[row["row_id"]]
            )
        elif v57_by_id is not None:
            risk_probability = (
                (1.0-v57_risk_weight)*risk_probability
                + v57_risk_weight*v57_by_id[str(row["row_id"])]["probability"]
            )
        if v18_calibration is not None:
            risk_parameters = v36_risk_calibration or v18_calibration
            lexical_weight = float(risk_parameters["lexical_weight"])
            risk_probability = (
                (1.0-lexical_weight)*risk_probability
                + lexical_weight*v18_lexical_by_id[str(row["row_id"])]
            )
            risk_logits = np.log(np.clip(risk_probability, 1e-8, 1.0))
            risk_logits[config.RISK_LABELS["Indicator"]] += float(
                risk_parameters.get("indicator_bias", 0.0))
            risk_logits[config.RISK_LABELS["Behavior"]] += float(
                risk_parameters.get("behavior_bias", 0.0))
            risk_logits[config.RISK_LABELS["Attempt"]] += float(
                risk_parameters["attempt_bias"])
            risk_probability = np.exp(risk_logits - risk_logits.max())
        elif tfidf_by_id is not None:
            risk_probability = (
                (1.0-tfidf_weight)*risk_probability
                + tfidf_weight*tfidf_by_id[str(row["row_id"])]
            )
        # V58 was strictly tested as a residual expert after the complete
        # Transformer/ordinal/counterfactual/lexical calibration above.
        if v58_by_id is not None:
            risk_probability = risk_probability / np.clip(
                np.sum(risk_probability), 1e-8, None
            )
            risk_probability = (
                (1.0-v58_weight)*risk_probability
                + v58_weight*v58_by_id[str(row["row_id"])]
            )
        if cv_probability_by_id is not None and cv_use_risk:
            risk_probability = (
                (1.0-cv_weight)*risk_probability
                + cv_weight*cv_probability_by_id[str(row["row_id"])]
            )
        risk_probability = risk_probability / np.clip(
            np.sum(risk_probability), 1e-8, None
        )
        if qwen_blend_weight is not None:
            qwen_probability = qwen_by_id[str(row["row_id"])]
            risk_probability = (
                (1.0-qwen_blend_weight)*risk_probability
                + qwen_blend_weight*qwen_probability
            )
            if qwen_class_bias is not None:
                risk_logits = np.log(np.clip(risk_probability, 1e-8, 1.0))
                risk_logits += qwen_class_bias
                risk_probability = np.exp(risk_logits-risk_logits.max())
        risk_id = int(np.argmax(risk_probability))
        if qwen_by_id is not None and qwen_blend_weight is None:
            qwen_probability = qwen_by_id[str(row["row_id"])]
            qwen_risk = int(np.argmax(qwen_probability))
            if (float(np.max(qwen_probability)) >= qwen_confidence_threshold
                    and qwen_risk != risk_id):
                risk_id = qwen_risk
                qwen_changes += 1
        evidence = row.pop("_evidence")
        if v18_calibration is not None:
            risk_id = correct_risk_only(row["_text"], risk_id)
            parameter_source = (v35_parameters if v35_parameters is not None else
                                v18_calibration["evidence_parameters_by_predicted_risk"])
            parameters = parameter_source[config.ID2RISK[risk_id]]
            seed2 = v18_seed2_by_id[str(row["row_id"])]
            seed_weight = float(v18_calibration["seed2_evidence_weight"])
            if v57_by_id is not None:
                v57 = v57_by_id[str(row["row_id"])]
                start = ((1.0-seed_weight-v57_evidence_weight)*row["_start_logits"]
                         + seed_weight*seed2["start"]
                         + v57_evidence_weight*v57["start"])
                end = ((1.0-seed_weight-v57_evidence_weight)*row["_end_logits"]
                       + seed_weight*seed2["end"]
                       + v57_evidence_weight*v57["end"])
            else:
                start = (1.0-seed_weight)*row["_start_logits"] + seed_weight*seed2["start"]
                end = (1.0-seed_weight)*row["_end_logits"] + seed_weight*seed2["end"]
            spans = decode_model_evidence(
                row["_text"], row["_offsets"], start, end,
                threshold=float(parameters["threshold"]),
                max_tokens=int(parameters["max_tokens"]),
                end_policy=str(parameters["end_policy"]), limit=5,
            )
            evidence = apply_evidence_policy(
                row["_text"], risk_id, spans,
                policy=str(parameters["cue_policy"]), topk=int(parameters["topk"]),
            )
        elif cv_probability_by_id is not None and cv_use_evidence:
            evidence = cv_evidence_by_id[str(row["row_id"])]
        cv_evidence_active = cv_probability_by_id is not None and cv_use_evidence
        if v18_calibration is not None:
            # V18 already applied its label-conditional cue policy above.
            # Re-applying the legacy global policy would silently change the
            # strict-tested evidence set (especially Ideation/Behavior top-k).
            pass
        elif evidence_v4 is not None and not cv_evidence_active:
            risk_id = correct_risk_only(row["_text"], risk_id)
            evidence = apply_evidence_policy(
                row["_text"], risk_id, evidence,
                policy=str(evidence_v4["cue_policy"]),
                topk=int(evidence_v4["topk"]),
            )
        else:
            risk_id, evidence = _apply_task1_rules(row["_text"], risk_id, evidence)
        if cv_evidence_active:
            evidence = evidence[:int(cv_calibration.get(
                "topk", config.TASK1_CV_MAX_EVIDENCE_PHRASES
            ))]
        # V63 is a separately gated correction for the small Indicator
        # subtype expressed through negated suicidal language.  It runs after
        # all risk/evidence ensembles so its strict test matches production.
        risk_id, evidence, polarity_changed = apply_polarity_correction(
            row["_text"], risk_id, evidence,
        )
        polarity_changes += int(polarity_changed)
        row["risk_level"] = config.ID2RISK[risk_id]
        row["evidence"] = "; ".join(evidence)
        row.pop("_offsets")
        row.pop("_start_logits")
        row.pop("_end_logits")
    if polarity_changes:
        print(f"Task 1 V63 negation-polarity corrections: {polarity_changes}")
    if qwen_by_id is not None and qwen_blend_weight is None:
        print(f"Task 1 Qwen3-8B gated risk changes: {qwen_changes}/{len(raw_rows)}")
    from inference.factor_predictor import standalone_factor_probabilities
    factor_row_ids, standalone_probs = standalone_factor_probabilities()
    standalone_used = standalone_probs is not None
    if standalone_used:
        probability_by_id = dict(zip(factor_row_ids, standalone_probs))
        all_factor_probs = np.vstack([probability_by_id[row["row_id"]] for row in raw_rows])
        print("Using standalone MentalRoBERTa probabilities for Task 2")
    else:
        all_factor_probs = np.vstack(all_factor_probs)
    factor_semantic_probabilities = all_factor_probs.copy()
    if task2_experiment == "v38":
        from inference.factor_definition_retrieval_v38 import v38_factor_probabilities
        v38_probabilities, replacement_weight = v38_factor_probabilities(
            [row["row_id"] for row in raw_rows]
        )
        all_factor_probs = (
            (1.0 - replacement_weight) * all_factor_probs
            + replacement_weight * v38_probabilities
        )
        factor_semantic_probabilities = all_factor_probs.copy()
        print(
            "Task 2 V38 experimental semantic replacement: "
            f"accepted MentalRoBERTa={1.0-replacement_weight:.2f}, V38={replacement_weight:.2f}",
            flush=True,
        )
    # Rank-based decoding is invariant to the absolute calibration scale. A
    # leak-free strict ablation showed that sparse word/character cues and
    # MentalRoBERTa semantics are strongly complementary.
    factor_texts = [row["_text"] for row in raw_rows]
    factor_cpu_probabilities = cpu_factor_probabilities(factor_texts)
    custom_cpu_expert_enabled = False
    from inference.case_syntax_v66 import case_syntax_factor_probabilities
    v66_cpu = case_syntax_factor_probabilities(factor_texts)
    if v66_cpu is not None:
        factor_cpu_probabilities = v66_cpu
        custom_cpu_expert_enabled = True
        print("Task 2 V66 case-sensitive syntax sparse expert enabled")
    from inference.syntax_aux_v68 import syntax_factor_probabilities
    v68_cpu = syntax_factor_probabilities(factor_texts)
    if v68_cpu is not None:
        factor_cpu_probabilities = v68_cpu
        custom_cpu_expert_enabled = True
        print("Task 2 V68 explicit syntax sparse expert enabled")
    from inference.factor_dedup_occurrence_v65 import dedup_occurrence_probabilities
    v65_cpu = dedup_occurrence_probabilities(factor_texts)
    if v65_cpu is not None:
        factor_cpu_probabilities = v65_cpu
        custom_cpu_expert_enabled = True
        print("Task 2 V65 deduplicated occurrence-aware sparse ensemble enabled")

    if custom_cpu_expert_enabled:
        all_factor_probs = (
            config.FACTOR_SEMANTIC_MODEL_WEIGHT * all_factor_probs
            + config.FACTOR_CPU_ENSEMBLE_WEIGHT * factor_cpu_probabilities
        )
    else:
        all_factor_probs = blend_cpu_factor_probabilities(factor_texts, all_factor_probs)
    factor_base_probabilities = all_factor_probs.copy()
    thresholds, prevalence, floor_ratio, empty_rate = load_factor_calibration()
    from inference.factor_cross_encoder import cross_encoder_probabilities
    cross_probabilities = cross_encoder_probabilities(
        [row["_text"] for row in raw_rows], [row["row_id"] for row in raw_rows]
    )
    prototype_probabilities, prototype_calibration = (None, None)
    if cross_probabilities is not None:
        from inference.factor_cross_encoder_v2 import prototype_cross_encoder_probabilities
        prototype_probabilities, prototype_calibration = prototype_cross_encoder_probabilities(
            [row["_text"] for row in raw_rows], [row["row_id"] for row in raw_rows]
        )
    if prototype_probabilities is not None:
        all_factor_probs = (
            float(prototype_calibration["base_weight"]) * factor_base_probabilities
            + float(prototype_calibration["old_cross_weight"]) * cross_probabilities
            + float(prototype_calibration["new_cross_weight"]) * prototype_probabilities
        )
        from inference.factor_signed_graph_stack_v21 import signed_graph_stack_probabilities
        v21_probability, v21_prevalence, v21_ratio = signed_graph_stack_probabilities(
            factor_texts, all_factor_probs, factor_semantic_probabilities,
            factor_cpu_probabilities, cross_probabilities, prototype_probabilities,
        )
        v21_used = v21_probability is not None
        if v21_used:
            all_factor_probs = v21_probability
            prevalence = v21_prevalence
            print(
                "Task 2 V21 signed label-dependency stack: "
                f"existing=0.90, adaptive graph/meta=0.10, ratio={v21_ratio:.2f}"
            )
        # MHLAT-v4 is a residual expert trained from the five one-hop
        # MentalRoBERTa folds. It can affect a submission only after its nested
        # user-disjoint calibration explicitly adopts it.
        from inference.factor_mhlat_v4 import mhlat_factor_probabilities
        mhlat_row_ids, mhlat_probabilities, mhlat_calibration = mhlat_factor_probabilities()
        if mhlat_probabilities is not None:
            mhlat_by_id = {
                str(row_id): probability
                for row_id, probability in zip(mhlat_row_ids, mhlat_probabilities)
            }
            ordered_mhlat = np.vstack([
                mhlat_by_id[str(row["row_id"])] for row in raw_rows
            ])
            mhlat_weight = float(mhlat_calibration["mhlat_weight"])
            all_factor_probs = (
                (1.0 - mhlat_weight) * all_factor_probs
                + mhlat_weight * ordered_mhlat
            )
            prevalence_ratio = float(mhlat_calibration["prevalence_ratio"])
            print(
                f"Task 2 MHLAT-v4 residual blend: accepted-v3="
                f"{1.0-mhlat_weight:.2f}, MHLAT-v4={mhlat_weight:.2f}"
            )
        else:
            prevalence_ratio = (v21_ratio if v21_used else
                                float(prototype_calibration["prevalence_ratio"]))
        from inference.factor_boundary_lexicon_v50 import apply_boundary_lexicon
        all_factor_probs, v50_used = apply_boundary_lexicon(factor_texts, all_factor_probs)
        if v50_used:
            print("Task 2 V50 definition-boundary correction enabled for 3 rare factors")
        from inference.factor_meaning_boundary_v54 import apply_meaning_boundary_v54
        all_factor_probs, v54_used = apply_meaning_boundary_v54(
            factor_texts, all_factor_probs
        )
        if v54_used:
            print("Task 2 V54 graded meaning-in-life correction enabled")
        factor_predictions = apply_prior_topk(
            all_factor_probs, prevalence,
            ratio=prevalence_ratio,
        )
        effective_thresholds = np.full(config.NUM_FACTORS, np.nan)
        print(
            "Task 2 decoding: base/old-cross/prototype-cross = "
            f"{prototype_calibration['base_weight']:.2f}/"
            f"{prototype_calibration['old_cross_weight']:.2f}/"
            f"{prototype_calibration['new_cross_weight']:.2f}"
        )
    elif cross_probabilities is not None:
        all_factor_probs = (
            config.FACTOR_CROSS_BASE_WEIGHT * factor_base_probabilities
            + config.FACTOR_CROSS_WEIGHT * cross_probabilities
        )
        factor_predictions = apply_prior_topk(
            all_factor_probs, prevalence, ratio=config.FACTOR_CROSS_TOPK_RATIO
        )
        effective_thresholds = np.full(config.NUM_FACTORS, np.nan)
        print(
            "Task 2 decoding: MentalRoBERTa/TF-IDF base + shared label cross-encoder "
            f"= {config.FACTOR_CROSS_BASE_WEIGHT:.2f}/"
            f"{config.FACTOR_CROSS_WEIGHT:.2f}, prevalence-ranked top-k "
            f"(ratio={config.FACTOR_CROSS_TOPK_RATIO:.2f})"
        )
    elif standalone_used:
        factor_predictions = apply_prior_topk(
            all_factor_probs, prevalence, ratio=config.FACTOR_TOPK_RATIO
        )
        effective_thresholds = np.full(config.NUM_FACTORS, np.nan)
        print(
            "Task 2 decoding: 5-fold MentalRoBERTa/TF-IDF "
            f"= {config.FACTOR_SEMANTIC_MODEL_WEIGHT:.2f}/"
            f"{config.FACTOR_CPU_ENSEMBLE_WEIGHT:.2f}, prevalence-ranked "
            f"top-k (ratio={config.FACTOR_TOPK_RATIO:.2f})"
        )
    else:
        factor_predictions, effective_thresholds = apply_calibrated_thresholds(
            all_factor_probs, thresholds, prevalence, floor_ratio, empty_rate
        )
    baseline_factor_predictions = factor_predictions.copy()
    if task2_experiment == "v69":
        from inference.factor_targeted_repair_v69 import apply_targeted_v69
        factor_predictions, v69_used = apply_targeted_v69(
            all_factor_probs, factor_semantic_probabilities,
            factor_predictions, texts=factor_texts, force=True,
        )
        if v69_used:
            print("Task 2 V69 targeted repair enabled for exactly 2 weak labels")
    if task2_experiment == "v70":
        from inference.factor_aligned_decoder_v70 import apply_v70
        factor_predictions, v70_used = apply_v70(
            all_factor_probs, factor_predictions, texts=factor_texts, force=True,
        )
        if v70_used:
            print("Task 2 V70 fold-aligned decoder enabled for exactly 2 labels")
    for row in raw_rows:
        row.pop("_text")
    rows = []
    for row, predictions in zip(raw_rows, factor_predictions):
        row["factors"] = str([
            config.ID2FACTOR[j] for j, value in enumerate(predictions) if value
        ])
        rows.append(row)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    experiment_outputs = {
        "v69": config.OUTPUT_DIR / "panda_targeted_v69.csv",
        "v70": config.OUTPUT_DIR / "panda_targeted_v70.csv",
    }
    if task1_experiment == "qwen3-8b":
        output = config.OUTPUT_DIR / "panda_qwen3_8b_gate.csv"
    elif task1_experiment == "qwen3-8b-blend":
        output = config.OUTPUT_DIR / "panda_qwen3_8b_blend.csv"
    elif task1_experiment == "qwen3-8b-attempt-calibrated":
        output = config.OUTPUT_DIR / "panda_qwen3_8b_attempt_calibrated.csv"
    else:
        output = experiment_outputs.get(task2_experiment, config.OUTPUT_DIR / "panda.csv")
    result_frame = pd.DataFrame(rows)
    result_frame.to_csv(output, index=False)
    if task2_experiment in ("v69", "v70"):
        # Keep panda.csv on the official-best Task-2 baseline.  The local V69
        # candidate is deliberately a separate file until the leaderboard
        # confirms that its two label-local changes transfer.
        restored = result_frame.copy()
        restored["factors"] = [
            str([config.ID2FACTOR[j] for j, value in enumerate(prediction) if value])
            for prediction in baseline_factor_predictions
        ]
        restored.to_csv(config.OUTPUT_DIR / "panda.csv", index=False)
        print(f"Restored official-best baseline: {config.OUTPUT_DIR / 'panda.csv'}")
    empty = int((factor_predictions.sum(axis=1) == 0).sum())
    print(f"Task 2 predictions: mean={factor_predictions.sum(axis=1).mean():.2f} factors/post, "
          f"empty={empty}/{len(factor_predictions)}")
    if not standalone_used:
        print("Effective factor thresholds:", np.round(effective_thresholds, 3).tolist())
    print(f"Saved submission: {output}")


if __name__ == "__main__":
    predict()
