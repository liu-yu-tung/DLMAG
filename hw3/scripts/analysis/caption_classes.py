import argparse
import json
import re
from pathlib import Path

import pandas as pd

HW3 = Path(__file__).resolve().parents[2]
MODEL = "Qwen/Qwen3-4B-Instruct-2507"
BOOLEANS = {
    "asks_for_vocals": "the caption says singing, rapping, a voice or a choir is present in the music",
    "says_no_vocals": "the caption says the music is instrumental or has no vocals",
    "mentions_recording_quality": "the caption comments on recording or audio quality (low quality, noisy, amateur, live recording, lo-fi as a fault)",
    "names_genre": "the caption names a musical genre or style",
    "mentions_tempo_or_rhythm": "the caption describes tempo, beat or rhythm",
    "mentions_non_music_sounds": "the caption mentions speech, crowd noise, sound effects or other non-musical sounds",
    "gives_use_context": "the caption says where or when the music could be used or heard (a film scene, a club, a game, a ceremony)",
    "mostly_mood_words": "the caption is mainly mood or atmosphere words, with at most one concrete musical element",
}
SYSTEM = ("You label music captions. Answer with one JSON object and nothing else. Keys: "
          + "; ".join(f'"{key}" (true or false): {meaning}' for key, meaning in BOOLEANS.items())
          + '; "instrument_count" (integer): how many different instruments or sound sources the caption names, voices not counted.')


def parse(text: str) -> dict:
    found = re.search(r"\{.*\}", text, flags=re.S)
    record = json.loads(found.group(0))
    row = {key: bool(record[key]) for key in BOOLEANS}
    row["instrument_count"] = int(record["instrument_count"])
    return row


def main():
    parser = argparse.ArgumentParser(description="Label each caption with fixed content classes using the tag-extraction model (greedy). "
                                                 "Writes only booleans and counts per id, no caption text. Run with external/MQAScore/.venv-mf.")
    parser.add_argument("--set", dest="set_name", required=True, choices=["dev", "test"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--self-test", action="store_true", help="check the parser on a canned answer, no model")
    args = parser.parse_args()
    if args.self_test:
        canned = "```json\n{" + ", ".join(f'"{key}": true' for key in BOOLEANS) + ', "instrument_count": 3}\n```'
        assert parse(canned) == {**{key: True for key in BOOLEANS}, "instrument_count": 3}
        print("parser ok;", len(SYSTEM), "characters of instructions")
        return

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    captions = [json.loads(line) for line in open(HW3 / f"outputs/mqa/captions_{args.set_name}.jsonl")][: args.limit]
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    rows, failures = [], []
    for caption in captions:
        conversation = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": caption["text"]}]
        ids = tokenizer.apply_chat_template(conversation, add_generation_prompt=True, return_tensors="pt", return_dict=True).to("cuda")
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=200, do_sample=False)
        answer = tokenizer.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)
        try:
            rows.append({"id": str(caption["id"]), **parse(answer), "words": len(caption["text"].split())})
        except (AttributeError, KeyError, ValueError):
            failures.append(str(caption["id"]))
    out_path = args.out or HW3 / f"results/analysis/caption_classes_{args.set_name}.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"{len(rows)} labelled, {len(failures)} unparsed {failures} -> {out_path}")
    assert len(failures) <= 0.05 * len(captions), "too many unparsed answers"


if __name__ == "__main__":
    main()
