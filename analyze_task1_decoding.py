"""Scan evidence decoding settings on the strict outer user holdout."""
import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Subset

from configs.config import config
from datasets.dataset import SuicideRiskDataset
from datasets.collator import SuicideRiskCollator
from models.multitask_model import SuicideRiskMultiTaskModel
from baseline import _apply_task1_rules, _post_phrase_f1
from utils.task1_metric import task1_score as competition_task1_score


def decode(text, offsets, start_logits, end_logits, threshold, max_tokens):
    start = torch.sigmoid(start_logits).cpu().numpy()
    end = torch.sigmoid(end_logits).cpu().numpy()
    candidates = []
    for chunk, chunk_offsets in enumerate(offsets):
        starts = np.where(start[chunk] >= threshold)[0]
        ends = np.where(end[chunk] >= threshold)[0]
        for s in starts:
            valid = [e for e in ends if s <= e <= s + max_tokens and chunk_offsets[s][1] > chunk_offsets[s][0]]
            if valid:
                e = min(valid, key=lambda x: x - s)
                a, b = chunk_offsets[s][0], chunk_offsets[e][1]
                phrase = text[a:b].strip()
                if phrase:
                    candidates.append((float(start[chunk, s] * end[chunk, e]), phrase))
    selected = []
    for _, phrase in sorted(candidates, reverse=True):
        norm = " ".join(phrase.lower().split())
        if not any(norm in p.lower() or p.lower() in norm for p in selected):
            selected.append(phrase)
        if len(selected) == 5:
            break
    return selected


def main():
    dataset = SuicideRiskDataset(config.CACHE_DIR / "train_cache.pt")
    labels = np.asarray([int(x["risk_label"]) for x in dataset.data])
    groups = np.asarray([x["anon_user_id"] for x in dataset.data])
    folds = StratifiedGroupKFold(n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED)
    _, valid_idx = next(folds.split(np.zeros(len(dataset)), labels, groups))
    loader = DataLoader(Subset(dataset, valid_idx), batch_size=config.BATCH_SIZE, shuffle=False,
                        collate_fn=SuicideRiskCollator(), num_workers=0)
    device = torch.device(config.DEVICE)
    model = SuicideRiskMultiTaskModel().to(device)
    model.load_state_dict(torch.load(config.OUTPUT_DIR / "best_model.pt", map_location=device))
    model.eval(); examples = []
    with torch.no_grad():
        for batch in loader:
            output = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            risks = output["risk_logits"].argmax(-1).cpu().tolist()
            for i, raw_risk in enumerate(risks):
                examples.append((batch["texts"][i], batch["offset_mappings"][i],
                                 output["start_logits"][i].cpu(), output["end_logits"][i].cpu(),
                                 raw_risk, int(batch["risk_labels"][i]), batch["evidences"][i]))
    results = {}
    for threshold in (0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55):
        for max_tokens in (8, 16, 24, 40):
            truth, pred, phrase = [], [], []
            for text, offsets, start, end, raw_risk, gold_risk, gold_evidence in examples:
                evidence = decode(text, offsets, start, end, threshold, max_tokens)
                risk, evidence = _apply_task1_rules(text, raw_risk, evidence)
                truth.append(gold_risk); pred.append(risk)
                phrase.append(_post_phrase_f1(evidence, gold_evidence))
            risk_f1 = f1_score(truth, pred, average="weighted")
            phrase_f1 = float(np.mean(phrase))
            results[(threshold, max_tokens)] = (
                risk_f1, phrase_f1, competition_task1_score(risk_f1, phrase_f1)
            )
    for key, values in sorted(results.items(), key=lambda x: x[1][2], reverse=True)[:10]:
        print(f"threshold={key[0]:.2f} max_tokens={key[1]:2d} risk={values[0]:.6f} "
              f"phrase={values[1]:.6f} task1={values[2]:.6f}")


if __name__ == "__main__":
    main()
