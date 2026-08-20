"""Dedicated local translation pass for the bilingual review workbook."""
from __future__ import annotations

import json

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from configs.config import config


OUTPUT = config.OUTPUT_DIR / "factor_boundary_review_v19"
REVIEW_DATA = OUTPUT / "review_records.json"
PARTIAL = OUTPUT / "translations_zh.json"
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
BATCH_SIZE = 12
MAX_NEW_TOKENS = 1100
MANUAL_REPAIRS = {
    205: "我真想尖叫，因为我太懦弱，不敢自杀；我只想让一切结束。",
    206: "但我并没有尽全力。",
    207: "我可以就这样死去，也没什么，因为一切都一样。也许我就没那么害怕，可以摆脱对坏事的恐惧、努力追求好事；身边所有人都鄙视我，这很难受。我只是很害怕，每天看似完美……",
    208: "我只想躺在某个人怀里，为生存的恐惧大哭；一切都太难承受了。",
    507: "我再也做不到了。早上醒来就盼着夜晚，好回去睡觉。我想死，我讨厌自己。世界晚安。",
    508: "如果今晚没人查看我，我就结束生命。我会整晚把自己隔离起来，直到早上7点才回室友那里；如果没人注意到，我打算明天结束生命，因为我真的觉得没人会注意。",
    509: "我失业了，而且是我所爱每个人身上的寄生虫。",
    510: "我因为太想吸入剂而哭，却得撒谎说是在为上周被车撞死的猫哭。我讨厌自己，只想吸嗨后睡一整天，嗨到忘记吃饭然后死去。",
}


def _load():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    quantization = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=quantization, device_map={"": 0},
        local_files_only=True,
    )
    model.eval(); return model, tokenizer


def _prompt(batch):
    cases = "\n".join(f"{row['review_id']}: {row['sentence']}" for row in batch)
    return (
        "Faithfully translate each English Reddit sentence into natural, easy-to-understand "
        "Simplified Chinese. Preserve negation, uncertainty, first/third person, and whether "
        "support was received or only wanted. Translate the sentence meaning; never replace it "
        "with a category name such as 绝望、低自尊、接受 or 拒绝. Keep the emotional tone but "
        "do not add facts. Return exactly a JSON array of objects {\"id\": integer, "
        "\"zh\": \"translation\"}; no Markdown or explanation.\n\n" + cases
    )


def _parse(raw, expected):
    text = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    left, right = text.find("["), text.rfind("]")
    if left < 0 or right <= left:
        return None
    try:
        value = json.loads(text[left:right + 1])
    except json.JSONDecodeError:
        return None
    result = {}
    for item in value if isinstance(value, list) else []:
        try:
            identifier = int(item["id"]); translation = str(item["zh"]).strip()
        except (KeyError, TypeError, ValueError):
            continue
        if translation:
            result[identifier] = translation
    return result if all(identifier in result for identifier in expected) else None


@torch.inference_mode()
def translate(force=False):
    if not torch.cuda.is_available():
        raise RuntimeError("Dedicated translation requires CUDA")
    rows = json.loads(REVIEW_DATA.read_text(encoding="utf-8"))
    partial = {} if force or not PARTIAL.exists() else json.loads(PARTIAL.read_text(encoding="utf-8"))
    pending = [row for row in rows if str(row["review_id"]) not in partial]
    if pending:
        model, tokenizer = _load()
        for start in tqdm(range(0, len(pending), BATCH_SIZE), desc="Chinese translations"):
            batch = pending[start:start + BATCH_SIZE]
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": _prompt(batch)}], tokenize=False,
                add_generation_prompt=True,
            )
            encoded = tokenizer(rendered, return_tensors="pt", truncation=True,
                                max_length=6144).to("cuda")
            generated = model.generate(
                **encoded, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            raw = tokenizer.decode(generated[0, encoded.input_ids.shape[1]:],
                                   skip_special_tokens=True)
            expected = [row["review_id"] for row in batch]
            parsed = _parse(raw, expected)
            if parsed is None:
                # Leave the already generated review translation visible for
                # rare format failures instead of inventing an empty value.
                parsed = {row["review_id"]: row["chinese_meaning"] for row in batch}
            partial.update({str(key): value for key, value in parsed.items()})
            PARTIAL.write_text(json.dumps(partial, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"Translations completed: {len(partial)}/{len(rows)}", flush=True)
        del model; torch.cuda.empty_cache()
    for row in rows:
        row["chinese_meaning"] = MANUAL_REPAIRS.get(
            row["review_id"], partial[str(row["review_id"])]
        )
    REVIEW_DATA.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"translated": len(rows), "model": MODEL_NAME}, ensure_ascii=False), flush=True)
    return rows


def repair_fallbacks():
    """Retry only batches that fell back to the old category-like summary."""
    global BATCH_SIZE, MAX_NEW_TOKENS
    translations = json.loads(PARTIAL.read_text(encoding="utf-8"))
    judge = json.loads((OUTPUT / "qwen_reviews.json").read_text(encoding="utf-8"))
    failed = [key for key, value in translations.items()
              if value == str(judge.get(key, {}).get("zh", ""))]
    for key in failed:
        translations.pop(key, None)
    PARTIAL.write_text(json.dumps(translations, ensure_ascii=False, indent=2), encoding="utf-8")
    BATCH_SIZE = 4
    MAX_NEW_TOKENS = 500
    print(f"Retrying {len(failed)} fallback translations in smaller batches", flush=True)
    return translate()


if __name__ == "__main__":
    translate()
