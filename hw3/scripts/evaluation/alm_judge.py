"""Second audio-language judge: Qwen2-Audio-7B-Instruct (8-bit) answering MQAScore's per-tag yes/no questions.

Questions, system prompt, Yes/No token sets, P(Yes) formula and the item mean come from external/MQAScore
(prompts.py, scoring.py, tags.py), so the only change against MQAScore is the judge model. Batch size 1, last-position logits.
Run with HW1's venv: python -P scripts/evaluation/alm_judge.py --set dev --run cfg7 --audio-dir outputs/dev/cfg7 --tags outputs/mqa/tags_dev.jsonl
Writes results/<set>__<run>/alm_qwen_summary.json and alm_qwen_per_item.csv (suffix _limit<N> with --limit).
Resumable: scores append to outputs/alm_qwen/<set>__<run>/score.jsonl; the tracked files refresh every 20 items.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio

HW3_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HW3_ROOT / "external" / "MQAScore"))
from mqascore.io import append_jsonl, check_protocol, done_ids, read_jsonl  # noqa: E402
from mqascore.prompts import DEFAULT_ATTRIBUTES, pertag_question, question_template, system_prompt  # noqa: E402
from mqascore.scoring import NO_WORDS, YES_WORDS, pyes_from_logits, yes_no_ids  # noqa: E402
from mqascore.tags import normalize_tags  # noqa: E402

MODEL_ID = "Qwen/Qwen2-Audio-7B-Instruct"
SAMPLE_RATE = 16000
BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED = 1000, 0
SAVE_EVERY = 20


def build_conversation(question: str) -> list[dict]:
    return [
        {"role": "system", "content": [{"type": "text", "text": system_prompt()}]},
        {"role": "user", "content": [{"type": "audio", "audio_url": "clip"}, {"type": "text", "text": question}]},
    ]


def prompt_text(processor, question: str) -> str:
    return processor.apply_chat_template(build_conversation(question), add_generation_prompt=True, tokenize=False)


def load_wave(path: str, fix_seconds: float) -> np.ndarray:
    samples, rate = sf.read(path, dtype="float32", always_2d=True)
    mono = torch.from_numpy(samples.mean(axis=1))
    wave = torchaudio.functional.resample(mono, rate, SAMPLE_RATE).numpy() if rate != SAMPLE_RATE else mono.numpy()
    if fix_seconds:
        target = round(fix_seconds * SAMPLE_RATE)
        wave = np.pad(wave, (0, target - len(wave))) if len(wave) < target else wave[:target]
    return wave


def load_model(device: str):
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2AudioForConditionalGeneration

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    processor.tokenizer.padding_side = "left"
    quantization = BitsAndBytesConfig(load_in_8bit=True, llm_int8_skip_modules=["audio_tower", "multi_modal_projector", "lm_head"])
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        MODEL_ID, quantization_config=quantization, dtype=torch.float16, device_map={"": device}).eval()
    return processor, model


@torch.inference_mode()
def score_item(processor, model, wave: np.ndarray, tags: dict, attributes: list[str], yes_ids: torch.Tensor, no_ids: torch.Tensor) -> dict:
    flat = normalize_tags(tags, attributes)
    per_attribute: dict[str, list[dict]] = {}
    for attribute, tag, _ in flat:
        question = pertag_question(attribute, tag)
        inputs = processor(text=[prompt_text(processor, question)], audio=[wave], sampling_rate=SAMPLE_RATE,
                           return_tensors="pt", padding=True).to(model.device)
        inputs["input_features"] = inputs["input_features"].to(torch.float16)
        logits = model(**inputs).logits[:, -1, :]
        p_yes, loglik_yes, loglik_no = pyes_from_logits(logits, yes_ids.to(logits.device), no_ids.to(logits.device))
        per_attribute.setdefault(attribute, []).append({"tag": tag, "score": p_yes[0], "loglik_yes": loglik_yes[0], "loglik_no": loglik_no[0]})
    scores = [entry["score"] for entries in per_attribute.values() for entry in entries]
    return {
        "n_tags": len(scores),
        "mean_score": sum(scores) / len(scores) if scores else None,
        "attribute_mean": {a: sum(e["score"] for e in v) / len(v) for a, v in per_attribute.items()},
        "per_attribute": per_attribute,
    }


def bootstrap_ci(values) -> list[float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    means = values[rng.integers(0, len(values), size=(BOOTSTRAP_RESAMPLES, len(values)))].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def write_outputs(score_path: Path, out_dir: Path, suffix: str, attributes: list[str], n_expected: int) -> dict:
    rows = read_jsonl(score_path)
    per_item = pd.DataFrame([
        {"id": r["id"], "n_tags": r["n_tags"], "alm_qwen": r["mean_score"],
         **{f"{a}_mean": r["attribute_mean"].get(a) for a in attributes},
         **{f"{a}_n": len(r["per_attribute"].get(a, [])) for a in attributes}}
        for r in rows])
    per_item = per_item.sort_values("id", key=lambda s: s.map(lambda x: (len(x), x)))
    per_item.to_csv(out_dir / f"alm_qwen_per_item{suffix}.csv", index=False)
    scored = per_item.dropna(subset=["alm_qwen"])
    summary = {
        "judge": MODEL_ID, "quantization": "8-bit LLM, fp16 audio tower, projector and lm_head",
        "partial": len(per_item) < n_expected,
        "n_items_expected": n_expected, "n_items_scored": len(per_item), "n_items_with_tags": len(scored),
        "n_questions": int(per_item["n_tags"].sum()),
        "alm_qwen_mean": float(scored["alm_qwen"].mean()) if len(scored) else None,
        "alm_qwen_ci95": bootstrap_ci(scored["alm_qwen"]) if len(scored) else None,
        "bootstrap": {"resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED, "unit": "items"},
        "attribute_mean": {}, "attribute_n_items": {}, "attribute_n_questions": {},
    }
    for a in attributes:
        present = per_item[f"{a}_mean"].dropna()
        summary["attribute_n_items"][a] = len(present)
        summary["attribute_n_questions"][a] = int(per_item[f"{a}_n"].sum())
        summary["attribute_mean"][a] = float(present.mean()) if len(present) else None
    (out_dir / f"alm_qwen_summary{suffix}.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def dry_run(items: list[dict], attributes: list[str], fix_seconds: float) -> None:
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    yes_list, no_list = yes_no_ids(processor.tokenizer)
    print("yes ids", yes_list, [processor.tokenizer.decode([i]) for i in yes_list])
    print("no ids", no_list, [processor.tokenizer.decode([i]) for i in no_list])
    assert len(yes_list) == len(YES_WORDS) and len(no_list) == len(NO_WORDS) and not set(yes_list) & set(no_list)
    for attribute in attributes:
        print(f"template {attribute}: {question_template(attribute)}")
    print("prompt:", repr(prompt_text(processor, pertag_question(attributes[0], "jazz"))))

    vocab = processor.tokenizer.vocab_size + len(processor.tokenizer.get_added_vocab())
    logits = torch.full((2, vocab), -20.0)
    logits[0, yes_list] = -2.0
    logits[0, no_list] = -4.0
    logits[1, yes_list] = -5.0
    logits[1, no_list] = -1.0
    p_yes, loglik_yes, loglik_no = pyes_from_logits(logits, torch.tensor(yes_list), torch.tensor(no_list))
    expected = [1 / (1 + np.exp(-2.0)), 1 / (1 + np.exp(4.0))]
    assert np.allclose(p_yes, expected, atol=1e-6), (p_yes, expected)
    assert p_yes[0] > 0.5 > p_yes[1]
    print("fake-logit P(Yes):", p_yes, "expected", expected)

    for item in items:
        flat = normalize_tags(item["tags"], attributes)
        wave = load_wave(item["audio"], fix_seconds)
        text = prompt_text(processor, pertag_question(flat[0][0], flat[0][1])) if flat else None
        n_tokens = audio_tokens = None
        if text:
            inputs = processor(text=[text], audio=[wave], sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
            n_tokens = int(inputs["input_ids"].shape[1])
            audio_tokens = int((inputs["input_ids"] == processor.tokenizer.convert_tokens_to_ids("<|AUDIO|>")).sum())
            assert processor.tokenizer.decode(inputs["input_ids"][0, -1:]) == "\n"
            print(f"id {item['id']}: {len(flat)} questions, {len(wave) / SAMPLE_RATE:.2f}s audio, {n_tokens} prompt tokens ({audio_tokens} audio), features {tuple(inputs['input_features'].shape)}")
        else:
            print(f"id {item['id']}: no questions")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--set", dest="set_name", required=True, choices=["dev", "test"])
    parser.add_argument("--run", required=True, help="run name; outputs go to results/<set>__<run>/ ('real' = dev real clips)")
    parser.add_argument("--audio-dir", required=True, help="directory with <id>.wav for every id in --tags")
    parser.add_argument("--tags", required=True, help="tags jsonl: {id, tags: {attribute: [...]}} per line")
    parser.add_argument("--attributes", default=",".join(DEFAULT_ATTRIBUTES))
    parser.add_argument("--fix-seconds", type=float, default=10.0, help="zero-pad or truncate the clip to this length; 0 keeps it as is")
    parser.add_argument("--limit", type=int, default=None, help="first N items only (smoke run; files get a _limit<N> suffix)")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--dry-run", action="store_true", help="build prompts, check Yes/No ids and the scoring math, no model load")
    args = parser.parse_args()

    attributes = [a.strip() for a in args.attributes.split(",") if a.strip()]
    items = []
    for row in read_jsonl(args.tags)[: args.limit]:
        audio = Path(args.audio_dir) / f"{row['id']}.wav"
        assert audio.exists(), f"missing audio {audio}"
        items.append({"id": str(row["id"]), "audio": str(audio.resolve()), "tags": row["tags"]})
    assert items and len({it["id"] for it in items}) == len(items)

    if args.dry_run:
        dry_run(items, attributes, args.fix_seconds)
        return
    assert args.device == "cuda", "8-bit loading needs a GPU; use --dry-run on CPU"
    suffix = "" if args.limit is None else f"_limit{args.limit}"
    out_dir = HW3_ROOT / "results" / f"{args.set_name}__{args.run}"
    work_dir = HW3_ROOT / "outputs" / "alm_qwen" / f"{args.set_name}__{args.run}{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    score_path = work_dir / "score.jsonl"
    protocol = {"model_id": MODEL_ID, "system_prompt": system_prompt(), "attributes": attributes, "fix_seconds": args.fix_seconds,
                "templates": {a: question_template(a) for a in attributes}, "tags": Path(args.tags).name}
    check_protocol(score_path, protocol)
    finished = done_ids(score_path)
    todo = [it for it in items if it["id"] not in finished]
    print(f"{len(items)} items, {len(finished)} already scored, {len(todo)} to go", flush=True)

    if todo:
        processor, model = load_model(args.device)
        yes_ids, no_ids = (torch.tensor(ids) for ids in yes_no_ids(processor.tokenizer))
        started = time.time()
        for count, item in enumerate(todo, 1):
            result = score_item(processor, model, load_wave(item["audio"], args.fix_seconds), item["tags"], attributes, yes_ids, no_ids)
            append_jsonl(score_path, {"id": item["id"], **result})
            if count % SAVE_EVERY == 0:
                write_outputs(score_path, out_dir, suffix, attributes, len(items))
                print(f"{count}/{len(todo)} items, {(time.time() - started) / count:.1f}s per item", flush=True)
        print(f"peak GPU {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB, {time.time() - started:.0f}s", flush=True)

    summary = write_outputs(score_path, out_dir, suffix, attributes, len(items))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
