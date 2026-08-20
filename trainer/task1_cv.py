"""Five-fold user-disjoint Task 1 training and cross-fitted model selection."""
from __future__ import annotations

import json
from collections import Counter

import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from baseline import _apply_task1_rules, _post_phrase_f1
from configs.config import config
from datasets.cache_builder import build_cache
from datasets.collator import SuicideRiskCollator
from datasets.dataset import SuicideRiskDataset
from inference.predict import decode_evidence
from inference.task1_span_decoder import candidate_phrases, decode_span_candidates
from models.losses import EvidenceLoss
from models.task1_joint_model import Task1JointModel, optimizer_parameters, ordinal_class_probabilities
from utils.seed import seed_everything
from utils.task1_metric import task1_score as competition_task1_score


OUTPUT_DIR = config.OUTPUT_DIR / "task1_cv"
OOF_FILE = OUTPUT_DIR / "oof_predictions.npz"
RESULT_FILE = OUTPUT_DIR / "cv_results.json"
CALIBRATION_FILE = OUTPUT_DIR / "calibration.json"
TRAINING_VERSION = "task1-rationale-v3"


def _fold_paths(fold):
    return OUTPUT_DIR / f"fold{fold}_model.pt", OUTPUT_DIR / f"fold{fold}_valid.npz"


def _loader(dataset, indices, shuffle):
    return DataLoader(
        Subset(dataset, indices), batch_size=config.BATCH_SIZE, shuffle=shuffle,
        collate_fn=SuicideRiskCollator(), num_workers=config.NUM_WORKERS,
        pin_memory=config.DEVICE == "cuda",
    )


def _move(batch, device):
    for key in ("input_ids", "attention_mask", "risk_labels", "start_labels", "end_labels", "token_labels"):
        batch[key] = batch[key].to(device)
    return batch


class JointTask1Loss(torch.nn.Module):
    def __init__(self, class_weights, ordinal_pos_weight, token_pos_weight):
        super().__init__()
        self.register_buffer("class_weights", class_weights)
        self.register_buffer("ordinal_pos_weight", ordinal_pos_weight)
        self.register_buffer("token_pos_weight", token_pos_weight)
        self.evidence = EvidenceLoss()

    def forward(self, output, batch):
        risk = torch.nn.functional.cross_entropy(
            output["risk_logits"], batch["risk_labels"], weight=self.class_weights
        )
        thresholds = torch.arange(3, device=batch["risk_labels"].device)
        ordinal_targets = (batch["risk_labels"].unsqueeze(1) > thresholds).float()
        ordinal = torch.nn.functional.binary_cross_entropy_with_logits(
            output["ordinal_logits"], ordinal_targets, pos_weight=self.ordinal_pos_weight
        )
        consistency = output["risk_logits"].sum() * 0.0
        if "risk_logits_rdrop" in output:
            risk_second = torch.nn.functional.cross_entropy(
                output["risk_logits_rdrop"], batch["risk_labels"], weight=self.class_weights
            )
            ordinal_second = torch.nn.functional.binary_cross_entropy_with_logits(
                output["ordinal_logits_rdrop"], ordinal_targets,
                pos_weight=self.ordinal_pos_weight,
            )
            risk = 0.5 * (risk + risk_second)
            ordinal = 0.5 * (ordinal + ordinal_second)
            first_log = torch.nn.functional.log_softmax(output["risk_logits"], dim=-1)
            second_log = torch.nn.functional.log_softmax(
                output["risk_logits_rdrop"], dim=-1
            )
            first_prob = first_log.exp(); second_prob = second_log.exp()
            symmetric_kl = 0.5 * (
                torch.nn.functional.kl_div(first_log, second_prob, reduction="batchmean")
                + torch.nn.functional.kl_div(second_log, first_prob, reduction="batchmean")
            )
            ordinal_consistency = torch.nn.functional.mse_loss(
                torch.sigmoid(output["ordinal_logits"]),
                torch.sigmoid(output["ordinal_logits_rdrop"]),
            )
            consistency = symmetric_kl + ordinal_consistency
        span = self.evidence(
            output["start_logits"], output["end_logits"],
            batch["start_labels"], batch["end_labels"], batch["attention_mask"],
        )
        mask = batch["attention_mask"].float(); token_target = batch["token_labels"].float()
        raw = torch.nn.functional.binary_cross_entropy_with_logits(
            output["token_logits"], token_target, reduction="none"
        )
        weights = torch.where(token_target > 0, self.token_pos_weight.to(raw.dtype), 1.0)
        token_bce = (raw * weights * mask).sum() / mask.sum().clamp_min(1.0)
        token_prob = torch.sigmoid(output["token_logits"]) * mask
        intersection = (token_prob * token_target).sum((1, 2))
        denominator = token_prob.sum((1, 2)) + token_target.sum((1, 2))
        token_dice = (1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
        token = 0.65 * token_bce + 0.35 * token_dice
        # Rationale regularisation derived from the ERASER desiderata: match
        # gold evidence transitions (coherence) and the number of evidence
        # tokens per window (anti-over/under-extraction).
        adjacent = mask[..., 1:] * mask[..., :-1]
        predicted_transition = (token_prob[..., 1:] - token_prob[..., :-1]).abs()
        gold_transition = (token_target[..., 1:] - token_target[..., :-1]).abs()
        transition_raw = torch.nn.functional.smooth_l1_loss(
            predicted_transition, gold_transition, reduction="none"
        )
        transition = (transition_raw * adjacent).sum() / adjacent.sum().clamp_min(1.0)
        valid_chunks = (mask.sum(-1) > 0).float()
        predicted_count = token_prob.sum(-1)
        gold_count = (token_target * mask).sum(-1)
        count_raw = torch.nn.functional.smooth_l1_loss(
            torch.log1p(predicted_count), torch.log1p(gold_count), reduction="none"
        )
        count = (count_raw * valid_chunks).sum() / valid_chunks.sum().clamp_min(1.0)
        total = (
            risk + 0.30 * ordinal + span + 0.20 * token
            + config.TASK1_HEAD_RDROP_WEIGHT * consistency
            + config.TASK1_EVIDENCE_TRANSITION_WEIGHT * transition
            + config.TASK1_EVIDENCE_COUNT_WEIGHT * count
        )
        return total


def _criterion(dataset, indices, labels, device):
    counts = np.bincount(labels[indices], minlength=config.NUM_RISK_CLASSES)
    class_weights = np.sqrt(len(indices) / np.maximum(counts, 1))
    class_weights = torch.tensor(class_weights / class_weights.mean(), dtype=torch.float, device=device)
    ordinal_targets = labels[indices, None] > np.arange(3)[None, :]
    ordinal_positive = ordinal_targets.sum(0)
    ordinal_weights = torch.tensor(
        np.sqrt((len(indices) - ordinal_positive) / np.maximum(ordinal_positive, 1)),
        dtype=torch.float, device=device,
    )
    token_positive = sum(float(dataset.data[int(i)]["token_labels"].sum()) for i in indices)
    token_total = sum(float(dataset.data[int(i)]["attention_mask"].sum()) for i in indices)
    token_weight = torch.tensor(
        min(15.0, max(3.0, np.sqrt((token_total - token_positive) / max(token_positive, 1.0)))),
        dtype=torch.float, device=device,
    )
    return JointTask1Loss(class_weights, ordinal_weights, token_weight).to(device)


@torch.no_grad()
def _collect(model, loader, device):
    model.eval(); records = []
    for batch in loader:
        metadata = batch; batch = _move(batch, device)
        output = model(batch["input_ids"], batch["attention_mask"])
        standard = torch.softmax(output["risk_logits"], -1).cpu().numpy()
        ordinal = ordinal_class_probabilities(output["ordinal_logits"]).cpu().numpy()
        for i in range(len(metadata["row_id"])):
            candidates = decode_span_candidates(
                metadata["texts"][i], metadata["offset_mappings"][i],
                output["start_logits"][i], output["end_logits"][i], output["token_logits"][i],
            )
            legacy = decode_evidence(
                metadata["texts"][i], metadata["offset_mappings"][i],
                output["start_logits"][i], output["end_logits"][i],
            )
            records.append({
                "row_id": metadata["row_id"][i], "text": metadata["texts"][i],
                "truth": int(batch["risk_labels"][i].cpu()), "gold": metadata["evidences"][i],
                "standard": standard[i], "ordinal": ordinal[i],
                "candidate": candidate_phrases(candidates, 5), "legacy": legacy,
            })
    return records


def _evaluate(records, ordinal_weight=0.25, evidence_mode="candidate", topk=3):
    truth, prediction, phrases = [], [], []
    for record in records:
        probability = ((1.0 - ordinal_weight) * record["standard"]
                       + ordinal_weight * record["ordinal"])
        risk = int(np.argmax(probability))
        evidence = list(record[evidence_mode])[:topk]
        risk, evidence = _apply_task1_rules(record["text"], risk, evidence)
        evidence = evidence[:topk]
        truth.append(record["truth"]); prediction.append(risk)
        phrases.append(_post_phrase_f1(evidence, record["gold"]))
    risk_f1 = float(f1_score(truth, prediction, average="weighted", zero_division=0))
    phrase_f1 = float(np.mean(phrases))
    return {"risk_f1": risk_f1, "phrase_f1": phrase_f1,
            "task1": competition_task1_score(risk_f1, phrase_f1),
            "risk_predictions": prediction, "phrase_scores": phrases}


def _save_records(path, records, valid_idx, summary):
    candidates = np.empty(len(records), dtype=object)
    candidates[:] = [item["candidate"] for item in records]
    legacy = np.empty(len(records), dtype=object)
    legacy[:] = [item["legacy"] for item in records]
    np.savez_compressed(
        path, valid_indices=np.asarray(valid_idx),
        standard=np.vstack([item["standard"] for item in records]),
        ordinal=np.vstack([item["ordinal"] for item in records]),
        candidate=candidates,
        legacy=legacy,
        summary=json.dumps(summary),
    )


def _train_fold(fold, train_idx, valid_idx, dataset, labels, device):
    checkpoint, prediction_file = _fold_paths(fold)
    if checkpoint.exists() and prediction_file.exists():
        saved = np.load(prediction_file, allow_pickle=True)
        summary = json.loads(str(saved["summary"]))
        if (np.array_equal(saved["valid_indices"], valid_idx)
                and summary.get("training_version") == TRAINING_VERSION):
            print(f"Task 1 fold {fold}: resumed")
            return summary
    seed_everything(config.SEED + 200 + fold)
    train_loader = _loader(dataset, train_idx, True); valid_loader = _loader(dataset, valid_idx, False)
    model = Task1JointModel().to(device)
    criterion = _criterion(dataset, train_idx, labels, device)
    optimizer = AdamW(optimizer_parameters(model), weight_decay=config.WEIGHT_DECAY)
    updates = int(np.ceil(len(train_loader) / config.GRADIENT_ACCUMULATION)) * config.EPOCHS
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, max(1, int(updates * config.WARMUP_RATIO)), updates
    )
    scaler = torch.amp.GradScaler("cuda", enabled=config.FP16)
    best = {"fold": fold, "task1": -1.0, "training_version": TRAINING_VERSION}
    for epoch in range(1, config.EPOCHS + 1):
        model.train(); optimizer.zero_grad(set_to_none=True); losses = []
        for step, batch in enumerate(tqdm(train_loader, desc=f"task1 fold {fold} epoch {epoch}"), 1):
            batch = _move(batch, device)
            with torch.autocast(device_type=device.type, enabled=config.FP16):
                loss = criterion(model(batch["input_ids"], batch["attention_mask"]), batch)
                loss = loss / config.GRADIENT_ACCUMULATION
            scaler.scale(loss).backward(); losses.append(float(loss.detach()) * config.GRADIENT_ACCUMULATION)
            if step % config.GRADIENT_ACCUMULATION == 0 or step == len(train_loader):
                scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
                old_scale = scaler.get_scale(); scaler.step(optimizer); scaler.update()
                if scaler.get_scale() >= old_scale:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
        records = _collect(model, valid_loader, device)
        metric = _evaluate(records, ordinal_weight=0.25, evidence_mode="candidate", topk=3)
        print(f"task1 fold={fold} epoch={epoch} loss={np.mean(losses):.4f} "
              f"risk={metric['risk_f1']:.4f} phrase={metric['phrase_f1']:.4f} task1={metric['task1']:.4f}")
        if metric["task1"] > best["task1"]:
            best = {"fold": fold, "epoch": epoch, "train_loss": float(np.mean(losses)),
                    "training_version": TRAINING_VERSION,
                    **{key: metric[key] for key in ("risk_f1", "phrase_f1", "task1")}}
            torch.save(model.state_dict(), checkpoint)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    records = _collect(model, valid_loader, device)
    _save_records(prediction_file, records, valid_idx, best)
    del model, optimizer, scheduler, train_loader, valid_loader
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return best


def _records_from_oof(dataset, folds):
    records = [None] * len(dataset)
    for fold, (_, valid_idx) in enumerate(folds):
        saved = np.load(_fold_paths(fold)[1], allow_pickle=True)
        for local, global_index in enumerate(valid_idx):
            item = dataset.data[int(global_index)]
            records[int(global_index)] = {
                "row_id": item["row_id"], "text": item["text"],
                "truth": int(item["risk_label"]), "gold": item["evidence"],
                "standard": saved["standard"][local], "ordinal": saved["ordinal"][local],
                "candidate": list(saved["candidate"][local]), "legacy": list(saved["legacy"][local]),
            }
    if any(item is None for item in records):
        raise ValueError("Incomplete Task 1 OOF records")
    return records


def _parameter_grid(records, indices):
    candidates = []
    subset = [records[int(i)] for i in indices]
    for weight in (0.0, 0.15, 0.25, 0.35, 0.50):
        for mode in ("candidate", "legacy"):
            for topk in (2, 3, 4, 5):
                metric = _evaluate(subset, weight, mode, topk)
                candidates.append({"ordinal_weight": weight, "evidence_mode": mode, "topk": topk,
                                   **{key: metric[key] for key in ("risk_f1", "phrase_f1", "task1")}})
    return sorted(candidates, key=lambda item: item["task1"], reverse=True)


def train_task1_cv(only_fold0=False):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    build_cache(train=True)
    dataset = SuicideRiskDataset(config.CACHE_DIR / "train_cache.pt")
    labels = np.asarray([int(item["risk_label"]) for item in dataset.data])
    groups = np.asarray([item["anon_user_id"] for item in dataset.data])
    folds = list(StratifiedGroupKFold(
        n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED
    ).split(np.zeros(len(dataset)), labels, groups))
    device = torch.device(config.DEVICE); summaries = []
    for fold, (train_idx, valid_idx) in enumerate(folds):
        summaries.append(_train_fold(fold, train_idx, valid_idx, dataset, labels, device))
        if only_fold0:
            result = {"mode": "fold0_ablation", "fold": summaries[0]}
            reference_file = config.OUTPUT_DIR / "task1_v2_ensemble_results.json"
            if reference_file.exists():
                reference = json.loads(reference_file.read_text(encoding="utf-8"))["best"]
                result["stable_reference_same_split"] = reference
                result["task1_delta"] = float(
                    summaries[0]["task1"] - reference["task1"]
                )
            (OUTPUT_DIR / "fold0_ablation.json").write_text(
                json.dumps(result, indent=2), encoding="utf-8"
            )
            print(json.dumps(result, indent=2))
            return result

    records = _records_from_oof(dataset, folds)
    full_grid = _parameter_grid(records, np.arange(len(records)))
    crossfit_risk = np.zeros(len(records), dtype=int); crossfit_phrase = np.zeros(len(records))
    parameters = []
    for fold, (fit, valid) in enumerate(folds):
        selected = _parameter_grid(records, fit)[0]
        metric = _evaluate(
            [records[int(i)] for i in valid], selected["ordinal_weight"],
            selected["evidence_mode"], selected["topk"],
        )
        crossfit_risk[valid] = metric["risk_predictions"]
        crossfit_phrase[valid] = metric["phrase_scores"]
        parameters.append({"fold": fold, **selected})
    crossfit_risk_f1 = float(f1_score(labels, crossfit_risk, average="weighted", zero_division=0))
    crossfit_phrase_f1 = float(crossfit_phrase.mean())
    crossfit_task1 = competition_task1_score(crossfit_risk_f1, crossfit_phrase_f1)

    ordinal_weight = float(np.median([item["ordinal_weight"] for item in parameters]))
    evidence_mode = Counter(item["evidence_mode"] for item in parameters).most_common(1)[0][0]
    topk = Counter(item["topk"] for item in parameters).most_common(1)[0][0]
    fixed_metric = _evaluate(records, ordinal_weight, evidence_mode, topk)
    # Gate the two Task 1 outputs independently.  Production can keep the
    # established evidence extractor while accepting a stronger risk model,
    # or vice versa; a weak branch must never be carried in by the other one.
    use_risk = crossfit_risk_f1 >= 0.785
    use_evidence = evidence_mode == "candidate" and crossfit_phrase_f1 >= 0.733
    adopted = use_risk or use_evidence
    calibration = {
        "training_version": TRAINING_VERSION,
        "adopted": adopted, "ordinal_weight": ordinal_weight,
        "evidence_mode": evidence_mode, "topk": int(topk),
        "test_weight": config.TASK1_CV_TEST_WEIGHT,
        "use_risk": use_risk,
        # Test-time fold voting is implemented for the boundary-aware decoder.
        # Keep legacy evidence on the established path if CV selected it.
        "use_evidence": use_evidence,
        "crossfit_risk_f1": crossfit_risk_f1,
        "crossfit_phrase_f1": crossfit_phrase_f1,
        "crossfit_task1": crossfit_task1,
        "fixed_oof": {key: fixed_metric[key] for key in ("risk_f1", "phrase_f1", "task1")},
    }
    result = {"folds": summaries, "crossfit_parameters": parameters,
              "calibration": calibration, "best_full_oof": full_grid[0], "top10": full_grid[:10]}
    RESULT_FILE.write_text(json.dumps(result, indent=2), encoding="utf-8")
    CALIBRATION_FILE.write_text(json.dumps(calibration, indent=2), encoding="utf-8")
    np.savez_compressed(
        OOF_FILE,
        standard=np.vstack([item["standard"] for item in records]),
        ordinal=np.vstack([item["ordinal"] for item in records]), labels=labels,
    )
    print(json.dumps(result, indent=2)); return result


if __name__ == "__main__":
    train_task1_cv()
