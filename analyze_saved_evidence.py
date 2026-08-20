"""Decoder-only checks on saved strict predictions (no model inference required)."""
import csv
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ANCHOR = re.compile(
    r"(?i)(suicid|kill|die|dying|dead|death|end (?:my|this) life|not wake|never wake|"
    r"overdose|hang|rope|gun|shoot|knife|razor|wrist|bleed|pills|poison|jump|drown|"
    r"attempt|sleep forever|stop existing)"
)


def split(value):
    return [x.strip() for x in str(value or "").split(";") if x.strip()]


def match(pred, gold):
    pred = " ".join(pred.casefold().split()); gold = " ".join(gold.casefold().split())
    return bool(pred and gold and (pred in gold or gold in pred)
                and len(pred.split()) <= 3 * max(1, len(gold.split())))


def score(predictions, targets):
    if not predictions and not targets:
        return 1.0
    edges = [[j for j, gold in enumerate(targets) if match(pred, gold)] for pred in predictions]
    assigned = {}
    def augment(i, seen):
        for j in edges[i]:
            if j in seen:
                continue
            seen.add(j)
            if j not in assigned or augment(assigned[j], seen):
                assigned[j] = i
                return True
        return False
    tp = sum(augment(i, set()) for i in range(len(predictions)))
    precision = tp / len(predictions) if predictions else 0.0
    recall = tp / len(targets) if targets else 0.0
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def main():
    files = [
        ROOT / "outputs" / "private_holdout_predictions.csv",
        ROOT / "outputs" / "private_user_holdout_predictions.csv",
        ROOT / "outputs" / "private_post_holdout_predictions.csv",
    ]
    result = {}
    for path in files:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        variants = {}
        for topk in (1, 2, 3, 4, 5):
            variants[f"top{topk}"] = sum(
                score(split(row["pred_evidence"])[:topk], split(row["gold_evidence"]))
                for row in rows
            ) / len(rows)
        for topk in (2, 3, 4, 5):
            variants[f"anchor_top{topk}"] = sum(
                score([x for x in split(row["pred_evidence"]) if ANCHOR.search(x)][:topk],
                      split(row["gold_evidence"]))
                for row in rows
            ) / len(rows)
        result[path.name] = variants
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
