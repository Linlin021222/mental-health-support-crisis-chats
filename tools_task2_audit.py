import ast
import collections
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
FACTORS = [
    "mental health issues", "physical health/characteristic", "substance use", "hopelessness",
    "emotion dysregulation", "low self-esteem", "poor school performance", "low socio-economic status",
    "interpersonal violence", "prior self-harm or suicidal thought/attempt", "poor social support",
    "interpersonal difficulty", "dysfunctional family", "exposure to others' suicide", "stressful life event",
    "traumatic experience", "cognitive deficits", "suicide means (with access)",
    "sexual orientation related issues", "social support", "coping strategy", "psychological capital",
    "sense of responsibility", "meaning in life",
]


def parse(value):
    if pd.isna(value):
        return []
    try:
        labels = ast.literal_eval(value) if isinstance(value, str) else [value]
    except (ValueError, SyntaxError):
        labels = [part.strip() for part in re.split(r"[;,]", str(value)) if part.strip()]
    return list(labels) if isinstance(labels, (list, tuple, set)) else [labels]


train = pd.read_excel(ROOT / "data" / "train.xlsx")
gold = collections.Counter()
bad = collections.Counter()
sizes = []
for value in train["factors"]:
    labels = sorted(set(str(x).strip() for x in parse(value)))
    sizes.append(len(labels))
    gold.update(labels)
    bad.update(label for label in labels if label not in set(FACTORS))

print("TRAIN", {"posts": len(train), "empty": sizes.count(0), "avg_labels": sum(sizes) / len(sizes)})
print("UNKNOWN", dict(bad))
for label in FACTORS:
    print(f"GOLD\t{label}\t{gold[label]}")

submission = ROOT / "outputs" / "panda.csv"
if submission.exists():
    pred = pd.read_csv(submission)
    counts = collections.Counter()
    pred_sizes = []
    for value in pred["factors"].fillna("[]"):
        labels = [str(x).strip() for x in parse(value)]
        pred_sizes.append(len(labels))
        counts.update(labels)
    print("PRED", {
        "posts": len(pred), "empty": pred_sizes.count(0),
        "empty_rate": pred_sizes.count(0) / len(pred_sizes),
        "avg_labels": sum(pred_sizes) / len(pred_sizes),
    })
    for label in FACTORS:
        print(f"PRED\t{label}\t{counts[label]}")
