"""Ablate whether Task1 V2 adds signal to the validated legacy model."""
import json
import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Subset

from baseline import _apply_task1_rules, _post_phrase_f1, _evidence_lexicon, _extract_evidence
from configs.config import config
from datasets.collator import SuicideRiskCollator
from datasets.dataset import SuicideRiskDataset
from inference.predict import decode_evidence
from models.multitask_model import SuicideRiskMultiTaskModel
from models.multitask_model_v2 import SuicideRiskMultiTaskModelV2, ordinal_class_probabilities
from preprocess.preprocess import load_train_data
from trainer.train_task1_v2 import decode_token_evidence
from utils.task1_metric import task1_score as competition_task1_score


@torch.no_grad()
def collect(model, dataset, indices, device, v2=False):
    loader = DataLoader(
        Subset(dataset, indices), batch_size=config.BATCH_SIZE, shuffle=False,
        collate_fn=SuicideRiskCollator(), num_workers=0,
    )
    result = []
    model.eval()
    for batch in loader:
        output = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        standard = torch.softmax(output["risk_logits"], -1).cpu().numpy()
        ordinal = ordinal_class_probabilities(output["ordinal_logits"]).cpu().numpy() if v2 else None
        for i in range(len(batch["row_id"])):
            result.append({
                "truth": int(batch["risk_labels"][i]), "text": batch["texts"][i],
                "offsets": batch["offset_mappings"][i], "gold": batch["evidences"][i],
                "standard": standard[i], "ordinal": None if ordinal is None else ordinal[i],
                "span": decode_evidence(
                    batch["texts"][i], batch["offset_mappings"][i],
                    output["start_logits"][i], output["end_logits"][i],
                ),
                "token_logits": None if not v2 else output["token_logits"][i].cpu(),
            })
    return result


def deduplicate(phrases):
    selected = []
    for phrase in phrases:
        normal = " ".join(phrase.casefold().split())
        if phrase and not any(normal in x.casefold() or x.casefold() in normal for x in selected):
            selected.append(phrase)
        if len(selected) == 5:
            break
    return selected


def main():
    dataset = SuicideRiskDataset(config.CACHE_DIR / "train_cache.pt")
    labels = np.asarray([int(x["risk_label"]) for x in dataset.data])
    groups = np.asarray([x["anon_user_id"] for x in dataset.data])
    folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=config.SEED)
    train, valid = next(folds.split(np.zeros(len(dataset)), labels, groups))
    frame = load_train_data()
    lexicon = _evidence_lexicon(frame.iloc[train].reset_index(drop=True))
    device = torch.device(config.DEVICE)
    old_model = SuicideRiskMultiTaskModel().to(device)
    old_model.load_state_dict(torch.load(config.OUTPUT_DIR / "best_model.pt", map_location=device))
    old = collect(old_model, dataset, valid, device)
    del old_model
    if device.type == "cuda": torch.cuda.empty_cache()
    v2_model = SuicideRiskMultiTaskModelV2().to(device)
    v2_model.load_state_dict(torch.load(config.OUTPUT_DIR / "task1_v2_strict_model.pt", map_location=device))
    v2 = collect(v2_model, dataset, valid, device, v2=True)
    results = []
    for v2_weight in np.linspace(0.0, 0.5, 6):
        for evidence_mode in (
            "old", "lexicon_old", "old_lexicon", "old_v2span", "old_v2token", "v2token_old"
        ):
            thresholds = (None,) if "token" not in evidence_mode else (0.4, 0.5, 0.6, 0.7)
            for token_threshold in thresholds:
                truth, pred, phrase = [], [], []
                for a, b in zip(old, v2):
                    v2_probability = 0.75 * b["standard"] + 0.25 * b["ordinal"]
                    probability = (1-v2_weight) * a["standard"] + v2_weight * v2_probability
                    risk = int(np.argmax(probability))
                    lexical = _extract_evidence(a["text"], risk, lexicon)
                    if evidence_mode == "old": evidence = a["span"]
                    elif evidence_mode == "lexicon_old": evidence = deduplicate(lexical + a["span"])
                    elif evidence_mode == "old_lexicon": evidence = deduplicate(a["span"] + lexical)
                    elif evidence_mode == "old_v2span": evidence = deduplicate(a["span"] + b["span"])
                    else:
                        token = decode_token_evidence(
                            b["text"], b["offsets"], b["token_logits"], token_threshold, 12
                        )
                        evidence = deduplicate(
                            a["span"] + token if evidence_mode == "old_v2token" else token + a["span"]
                        )
                    risk, evidence = _apply_task1_rules(a["text"], risk, evidence)
                    truth.append(a["truth"]); pred.append(risk)
                    phrase.append(_post_phrase_f1(evidence, a["gold"]))
                risk_f1 = f1_score(truth, pred, average="weighted", zero_division=0)
                phrase_f1 = float(np.mean(phrase))
                results.append({
                    "v2_weight": float(v2_weight), "evidence_mode": evidence_mode,
                    "token_threshold": token_threshold, "risk_f1": risk_f1,
                    "phrase_f1": phrase_f1,
                    "task1": competition_task1_score(risk_f1, phrase_f1),
                })
    results.sort(key=lambda x: x["task1"], reverse=True)
    print(json.dumps(results[:15], indent=2))
    (config.OUTPUT_DIR / "task1_v2_ensemble_results.json").write_text(
        json.dumps({"best": results[0], "top15": results[:15]}, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
