"""Bilingual, conservative review of ambiguous factor-evidence sentences.

This is not represented as human gold annotation.  A local Qwen2.5-7B judge
reads the sentence in neighbouring context, and a deliberately high-precision
rule set independently checks the four most problematic polarity boundaries.
Only confident, non-conflicting decisions are eligible for weak supervision;
all disagreements remain explicitly ``uncertain`` for human review.
"""
from __future__ import annotations

import json
from pathlib import Path
import re

import torch
from tqdm import tqdm

from configs.config import config
from preprocess.preprocess import load_train_data
from trainer.factor_sentence_evidence_v16 import _sentences
from trainer.task1_local_counterfactual_v56 import _load_model


OUTPUT = config.OUTPUT_DIR / "factor_boundary_review_v19"
SOURCE = config.OUTPUT_DIR / "factor_sentence_evidence_v16" / "fold0_pseudo_evidence.jsonl"
PARTIAL = OUTPUT / "qwen_reviews.json"
REVIEW_DATA = OUTPUT / "review_records.json"
SUMMARY = OUTPUT / "review_summary.json"
TARGET_LABELS = {
    "low self-esteem", "hopelessness", "social support", "poor social support",
}
LOW_SCORE, HIGH_SCORE = .15, .40
BATCH_SIZE = 8

# A few very short/noisy cases made the local model return malformed JSON even
# after a single-case retry.  Keep them auditable with conservative manual
# translations rather than silently dropping them from the bilingual sheet.
FALLBACKS = {
    17: ("我受够了，也厌倦了事事不如别人。", "疲惫与比较，不足以证明未来无望"),
    18: ("那个该死的女人。", "是对他人的辱骂，不是负面自我评价"),
    19: ("我受够了看普通人在教室后面社交。", "表现社交比较，但未明确缺乏支持"),
    121: ("我一直希望能重新来过……", "表达后悔，但未来绝望含义不充分"),
    122: ("她很可能没那意思，我又一次被吊着胃口。", "更像关系受挫，不直接证明缺乏支持"),
    123: ("只有想到死亡时我才感到解脱。", "把死亡视为唯一解脱，暗示无其他出路"),
    125: ("所有人都嘲笑你，活着毫无意义。", "体现被排斥，但支持缺失仍需上下文"),
    126: ("天生有无法改变的东西，是我的错吗？", "表达自责疑问，但无明确自我贬低"),
    128: ("不如年轻时自杀，别再延长这种糟糕生活。", "预期生活持续糟糕，具有绝望含义"),
    235: ("不敢相信我失去了你。", "关系丧失可能影响支持，但单句不充分"),
    237: ("我非常想你。", "想念某人不等于缺乏社会支持"),
    385: ("成绩很好，但仍想自杀；大学作业得了A4。", "没有表达负面自我价值评价"),
    387: ("我的时间很快就要到了。", "可能是风险时间线，不直接证明绝望"),
    388: ("同龄人都毕业了，而我像被遗忘、努力无人看见。", "被忽视不等于明确自我贬低"),
    390: ("同龄人都毕业了，而我像被系统遗忘。", "被遗忘和无人接住可支持缺乏支持"),
}


def _normalise(text):
    return " ".join(str(text).split()).strip()


def _context(text, target):
    sentences = _sentences(text)
    target_key = _normalise(target).casefold()
    index = next((i for i, sentence in enumerate(sentences)
                  if _normalise(sentence).casefold() == target_key), None)
    if index is None:
        # The splitter is deterministic, but retain a useful fallback if a
        # future preprocessing version changes whitespace handling.
        full = _normalise(text); position = full.casefold().find(target_key)
        if position >= 0:
            return full[max(0, position - 220):position + len(target) + 220]
        return full[:520]
    start, stop = max(0, index - 1), min(len(sentences), index + 2)
    return " ".join(sentences[start:stop])[:720]


def _rule_decision(label, sentence, context):
    """Return accept/reject/unknown using only high-precision surface cues."""
    s = _normalise(sentence).casefold()
    # Avoid treating another person's quoted self-description as the author;
    # difficult quotation cases are deliberately left to the contextual judge.
    first_person = bool(re.search(r"\b(i|i'm|im|me|my|myself)\b", s))
    if label == "low self-esteem":
        positive = re.search(
            r"\b(i(?:'m| am|m)?\s+(?:so\s+)?(?:worthless|useless|ugly|pathetic|"
            r"a failure|a burden|unlovable|unattractive|inferior|disgusting|stupid)|"
            r"hate myself|hate my (?:face|body|looks)|i (?:suck|am not good enough)|"
            r"no one (?:could|would|will) (?:love|want) me)\b", s,
        )
        if positive and first_person:
            return "accept"
        if re.search(r"\b(hopeless|no point|kill myself|want to die|no one to talk|lonely)\b", s):
            return "reject"
    elif label == "hopelessness":
        positive = re.search(
            r"\b(hopeless|no hope|nothing (?:will|can) (?:ever )?(?:change|improve)|"
            r"never (?:get|getting) better|no future|no way out|trapped|stuck forever|"
            r"what(?:'s| is) the point|(?:life|everything|it) is pointless|"
            r"things won(?:'t| not) get better|cannot go on|can't go on)\b", s,
        )
        if positive:
            return "accept"
        if re.fullmatch(r".{0,30}\b(i want to die|kill myself|end my life)\b.{0,30}", s):
            return "reject"
    elif label == "social support":
        absent = re.search(
            r"\b(no ?one|nobody|none of (?:my|them)|without anyone)\b.{0,55}"
            r"\b(care|help|listen|support|talk|there for me)\b|"
            r"\b(wish|need|want) (?:i had |for )?(?:someone|somebody|a friend)\b", s,
        )
        received = re.search(
            r"\b(?:my |a |the |he |she |they |someone |somebody |friend|friends|family|"
            r"partner|therapist|counsell?or|doctor|parents?|mom|dad).{0,45}"
            r"\b(helped|helps|listened|listens|supports?|supported|cares?|comforted|"
            r"was there for me|been there for me|gave me advice|checked on me)\b", s,
        )
        if received and not absent:
            return "accept"
        if absent:
            return "reject"
    elif label == "poor social support":
        absent = re.search(
            r"\b(no ?one|nobody|alone|lonely|isolated|abandoned|ignored|unsupported|"
            r"no friends|not a single friend)\b|"
            r"\b(?:no ?one|nobody).{0,55}\b(?:cares?|listens?|understands?|talk to|help)\b|"
            r"\b(?:need|wish|want).{0,30}\b(?:someone|somebody|a friend).{0,20}"
            r"\b(?:talk|care|listen|help)\b", s,
        )
        received = re.search(
            r"\b(?:he|she|they|my friend|my family|my partner|my therapist).{0,45}"
            r"\b(?:helped|listened|supported|cares|was there|been there)\b", s,
        )
        if absent and not received:
            return "accept"
        if received:
            return "reject"
    return "unknown"


def _training_eligibility(row):
    """High-precision subset for training; workbook suggestions stay broader."""
    if row["suggested_final"] != "accept" or row["qwen_confidence"] < .80:
        return False, "仅使用置信度≥0.80的接受样本"
    s = _normalise(row["sentence"]).casefold()
    label = row["factor"]
    if label == "low self-esteem":
        cue = re.search(
            r"\b(worthless|useless|pathetic|ugly|unattractive|failure|loser|burden|"
            r"unlovable|inferior|disgusting|coward|idiot|stupid|waste of space|"
            r"not good enough|hate myself|hate my (?:face|body|looks)|don't deserve|"
            r"do not deserve|lousy excuse for a human)\b", s,
        )
        return bool(cue), ("存在明确负面自我评价" if cue else "缺少明确负面自我评价词")
    if label == "hopelessness":
        cue = re.search(
            r"\b(hopeless|no hope|no point|pointless|no future|no way out|trapped|"
            r"nothing (?:will|can) (?:ever )?(?:change|improve)|never (?:get|gets|getting) better|"
            r"only (?:option|way|relief)|can't go on|cannot go on|nothing to live for|"
            r"prolonging this (?:shit|shitty) life|things won(?:'t| not) get better)\b", s,
        )
        return bool(cue), ("存在无出路或未来不改善表达" if cue else "缺少明确未来绝望边界")
    if label == "social support":
        absent = re.search(r"\b(no ?one|nobody|without anyone|doesn'?t help|don'?t help)\b", s)
        received = re.search(
            r"\b(helped|helps me|listened|listens to me|supported|supports me|cares about me|"
            r"was there for me|been there for me|paying my therapy|purchasing me medication|"
            r"gave me advice|checked on me|friends who (?:understand|feel the same))\b", s,
        )
        eligible = bool(received) and not bool(absent)
        return eligible, ("明确描述已经收到支持" if eligible else "未明确收到支持或存在否定")
    if label == "poor social support":
        received = re.search(r"\b(helped me|listened to me|supported me|was there for me)\b", s)
        absent = re.search(
            r"\b(no ?one|nobody|alone|lonely|no friends|forgotten|ignored|abandoned|"
            r"unsupported|isolated|slipped through the net|doesn'?t help|don'?t help|"
            r"won'?t care|don'?t care|someone please help|not a single friend)\b", s,
        )
        eligible = bool(absent) and not bool(received)
        return eligible, ("明确描述孤独、无人关心或帮助失败" if eligible else "支持缺失边界不够明确")
    return False, "非本轮目标标签"


def _load_queue():
    if not SOURCE.exists():
        raise FileNotFoundError(f"Missing sentence evidence source: {SOURCE}")
    frame = load_train_data().reset_index(drop=True)
    source_rows = [json.loads(line) for line in SOURCE.read_text(encoding="utf-8").splitlines()
                   if line.strip()]
    result, seen = [], set()
    for row in source_rows:
        label = row["factor"]
        if label not in TARGET_LABELS:
            continue
        for selected in row["selected"]:
            score = float(selected["semantic_score"])
            if not LOW_SCORE <= score <= HIGH_SCORE:
                continue
            key = (str(row["row_id"]), label, _normalise(selected["sentence"]).casefold())
            if key in seen:
                continue
            seen.add(key)
            index = int(row["row_index"]); sentence = _normalise(selected["sentence"])
            context = _context(frame.text.iloc[index], sentence)
            result.append({
                "review_id": len(result) + 1,
                "row_index": index,
                "row_id": str(row["row_id"]),
                "factor": label,
                "annotation_count": int(row["annotation_count"]),
                "semantic_score": score,
                "sentence": sentence,
                "context": context,
                "rule_decision": _rule_decision(label, sentence, context),
            })
    return result


def _prompt(batch):
    cases = "\n".join(
        f"CASE {row['review_id']}\nTARGET FACTOR: {row['factor']}\n"
        f"TARGET SENTENCE: {row['sentence']}\nNEIGHBOURING CONTEXT: {row['context']}"
        for row in batch
    )
    return (
        "You are reviewing sentence-level evidence for a suicide-factor research dataset. "
        "This is annotation, not clinical advice. Judge whether the TARGET SENTENCE, interpreted "
        "with its neighbouring context, is faithful evidence for the TARGET FACTOR. Do not accept "
        "a sentence merely because the full post has that label. Boundaries: low self-esteem is a "
        "negative evaluation of the author's worth, not only a bad future; hopelessness is belief "
        "that the future cannot improve/no way out, not only sadness or wanting death; social support "
        "requires support actually received; poor social support requires absent/failed support, "
        "loneliness, rejection or abandonment. Handle negation and whose experience is described.\n"
        "Return exactly a JSON array. Each object: {\"id\": integer, \"decision\": "
        "\"accept\"|\"reject\"|\"uncertain\", \"confidence\": number 0 to 1, "
        "\"zh\": concise Chinese meaning (max 35 Chinese characters), \"reason_zh\": concise "
        "Chinese reason (max 30 Chinese characters)}. No Markdown or extra text.\n\n" + cases
    )


def _parse(raw, expected_ids):
    text = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    left, right = text.find("["), text.rfind("]")
    if left < 0 or right <= left:
        return None
    try:
        value = json.loads(text[left:right + 1])
    except json.JSONDecodeError:
        return None
    parsed = {}
    for item in value if isinstance(value, list) else []:
        try:
            identifier = int(item["id"]); decision = str(item["decision"]).casefold()
            confidence = float(item["confidence"])
        except (KeyError, TypeError, ValueError):
            continue
        if decision not in {"accept", "reject", "uncertain"}:
            continue
        parsed[identifier] = {
            "decision": decision, "confidence": max(0., min(1., confidence)),
            "zh": str(item.get("zh", "")).strip()[:100],
            "reason_zh": str(item.get("reason_zh", "")).strip()[:100],
            "raw": raw,
        }
    return parsed if all(identifier in parsed for identifier in expected_ids) else None


@torch.inference_mode()
def generate_reviews(force=False):
    if not torch.cuda.is_available():
        raise RuntimeError("Boundary review requires CUDA for local Qwen2.5-7B")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    queue = _load_queue()
    partial = {} if force or not PARTIAL.exists() else json.loads(PARTIAL.read_text(encoding="utf-8"))
    pending = [row for row in queue if str(row["review_id"]) not in partial]
    if pending:
        model, tokenizer = _load_model()
        for start in tqdm(range(0, len(pending), BATCH_SIZE), desc="bilingual factor review"):
            batch = pending[start:start + BATCH_SIZE]
            prompt = _prompt(batch)
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False,
                add_generation_prompt=True,
            )
            encoded = tokenizer(rendered, return_tensors="pt", truncation=True,
                                max_length=6144).to("cuda")
            output = model.generate(
                **encoded, max_new_tokens=900, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            raw = tokenizer.decode(output[0, encoded.input_ids.shape[1]:],
                                   skip_special_tokens=True)
            expected = [row["review_id"] for row in batch]
            parsed = _parse(raw, expected)
            if parsed is None:
                # Retry each failed case separately; this avoids losing seven
                # good reviews because one long JSON batch was truncated.
                parsed = {}
                for row in batch:
                    single_prompt = _prompt([row])
                    single_rendered = tokenizer.apply_chat_template(
                        [{"role": "user", "content": single_prompt}], tokenize=False,
                        add_generation_prompt=True,
                    )
                    single = tokenizer(single_rendered, return_tensors="pt", truncation=True,
                                       max_length=4096).to("cuda")
                    generated = model.generate(
                        **single, max_new_tokens=180, do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                    single_raw = tokenizer.decode(
                        generated[0, single.input_ids.shape[1]:], skip_special_tokens=True,
                    )
                    item = _parse(single_raw, [row["review_id"]])
                    parsed[row["review_id"]] = (item or {row["review_id"]: {
                        "decision": "uncertain", "confidence": 0., "zh": "",
                        "reason_zh": "模型输出格式失败，需要人工复核", "raw": single_raw,
                    }})[row["review_id"]]
            for identifier, value in parsed.items():
                partial[str(identifier)] = value
            PARTIAL.write_text(json.dumps(partial, ensure_ascii=False, indent=2),
                               encoding="utf-8")
            print(f"Boundary reviews completed: {len(partial)}/{len(queue)}", flush=True)
        del model
        torch.cuda.empty_cache()

    counts = {"accept": 0, "reject": 0, "uncertain": 0}
    disagreements = 0
    for row in queue:
        llm = partial[str(row["review_id"])]
        if not llm.get("zh") and row["review_id"] in FALLBACKS:
            llm["zh"], llm["reason_zh"] = FALLBACKS[row["review_id"]]
        rule = row["rule_decision"]; decision = llm["decision"]
        conflict = rule in {"accept", "reject"} and decision != rule
        confident = float(llm["confidence"]) >= .75
        if conflict or not confident or decision == "uncertain":
            final = "uncertain"
        else:
            final = decision
        disagreements += int(conflict)
        row.update({
            "qwen_decision": decision, "qwen_confidence": float(llm["confidence"]),
            "chinese_meaning": llm["zh"], "chinese_reason": llm["reason_zh"],
            "rule_qwen_conflict": conflict, "suggested_final": final,
            # Editable workbook column is initially the conservative suggestion.
            "human_final": final,
        })
        eligible, eligibility_reason = _training_eligibility(row)
        row["training_eligible"] = eligible
        row["eligibility_reason"] = eligibility_reason
        counts[final] += 1
    REVIEW_DATA.write_text(json.dumps(queue, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "records": len(queue), "score_range": [LOW_SCORE, HIGH_SCORE],
        "labels": {label: sum(row["factor"] == label for row in queue)
                   for label in sorted(TARGET_LABELS)},
        "suggested_decisions": counts, "rule_qwen_conflicts": disagreements,
        "reviewer": "local Qwen2.5-7B plus conservative boundary rules",
        "human_gold": False,
        "training_eligible": sum(row["training_eligible"] for row in queue),
    }
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return queue, summary


if __name__ == "__main__":
    generate_reviews()
