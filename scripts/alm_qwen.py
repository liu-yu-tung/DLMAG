"""Zero-shot Qwen2-Audio-7B-Instruct on both tasks by label log-likelihood ranking.

For each clip and each candidate label, the model scores log p(answer | audio, question) where the answer is the
label text followed by <|im_end|>; the six scores are normalized into a distribution (top-3 = three best labels).
The LLM runs in 8-bit (bitsandbytes); the audio encoder and projector stay in fp16. Two prompts per task: a plain
question and a cue-guided one that names what to listen for. Resumable: partial results are saved every --save-every
clips. Output: features/{key}_alm_{prompt}.npz with logp (N, 6) summed answer log-probs, sample_id, split.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torchaudio
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2AudioForConditionalGeneration

from hw1.data import LABELS, load_audio, load_manifest

HW1 = Path(__file__).resolve().parents[1]
MODEL_ID = "Qwen/Qwen2-Audio-7B-Instruct"
SR_QWEN = 16000

ANSWERS = {
    "A": {l: l for l in LABELS["A"]},
    "B": {"US": "United States", "UK": "United Kingdom", "Brazil": "Brazil", "Spain": "Spain",
          "Germany": "Germany", "Italy": "Italy"},
}
PROMPTS = {
    "A": {
        "plain": "In which decade was this song released? Answer with one of: 1960s, 1970s, 1980s, 1990s, 2000s, 2010s.",
        "cues": ("Listen to the recording and production: audio fidelity, mixing and loudness, drum sound, synthesizers "
                 "or electric guitars, and vocal style. In which decade was this song released? Answer with one of: "
                 "1960s, 1970s, 1980s, 1990s, 2000s, 2010s."),
    },
    "B": {
        "plain": ("This song was released in the 1980s. In which country was it released? Answer with one of: "
                  "United States, United Kingdom, Brazil, Spain, Germany, Italy."),
        "cues": ("This song was released in the 1980s. Listen to the language of the lyrics, the accent and singing "
                 "style, and the musical genre. In which country was it released? Answer with one of: United States, "
                 "United Kingdom, Brazil, Spain, Germany, Italy."),
    },
}


def load_model(device: str = "cuda"):
    proc = AutoProcessor.from_pretrained(MODEL_ID)
    proc.tokenizer.padding_side = "left"
    q = BitsAndBytesConfig(load_in_8bit=True, llm_int8_skip_modules=["audio_tower", "multi_modal_projector", "lm_head"])
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        MODEL_ID, quantization_config=q, dtype=torch.float16, device_map={"": device}).eval()
    return proc, model


def prompt_text(proc, question: str) -> str:
    conv = [{"role": "user", "content": [{"type": "audio", "audio_url": "clip"}, {"type": "text", "text": question}]}]
    return proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)


@torch.inference_mode()
def score_clip(proc, model, wav16: np.ndarray, prefix: str, answers: list[str], batch: int = 2) -> np.ndarray:
    """Summed log-prob of each answer (+ <|im_end|>) after the prefix, scored `batch` answers per forward pass
    (full-vocabulary logits for every position do not fit next to other GPU jobs at batch 6)."""
    return np.concatenate([_score(proc, model, wav16, prefix, answers[i : i + batch]) for i in range(0, len(answers), batch)])


def _score(proc, model, wav16: np.ndarray, prefix: str, answers: list[str]) -> np.ndarray:
    tails = [a + "<|im_end|>" for a in answers]
    n_tail = [len(proc.tokenizer(t, add_special_tokens=False).input_ids) for t in tails]
    inp = proc(text=[prefix + t for t in tails], audio=[wav16] * len(tails), sampling_rate=SR_QWEN,
               return_tensors="pt", padding=True).to(model.device)
    inp["input_features"] = inp["input_features"].to(torch.float16)
    ids = inp["input_ids"]
    m = max(n_tail)
    logits = model(**inp, use_cache=False).logits[:, -m - 1 : -1].float()
    logp = torch.log_softmax(logits, -1).gather(-1, ids[:, -m:, None])[..., 0]
    return np.array([logp[i, m - n:].sum().item() for i, n in enumerate(n_tail)])


@torch.inference_mode()
def generate(proc, model, wav16: np.ndarray, question: str, max_new_tokens: int = 40) -> str:
    inp = proc(text=[prompt_text(proc, question)], audio=[wav16], sampling_rate=SR_QWEN, return_tensors="pt",
               padding=True).to(model.device)
    inp["input_features"] = inp["input_features"].to(torch.float16)
    out = model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False)
    return proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0]


def to16(y: np.ndarray) -> np.ndarray:
    return torchaudio.functional.resample(torch.as_tensor(y), 24000, SR_QWEN).numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["A", "B"])
    ap.add_argument("--prompts", nargs="+", default=["plain", "cues"])
    ap.add_argument("--splits", nargs="+", default=["validation"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--save-every", type=int, default=50)
    ap.add_argument("--smoke", action="store_true", help="also print free-form answers for the first clips")
    args = ap.parse_args()

    key = args.dataset
    df = load_manifest(key)
    df = df[df["split"].isin(args.splits)].reset_index(drop=True)
    if args.limit:
        df = df.groupby("split", group_keys=False).head(args.limit).reset_index(drop=True)
    proc, model = load_model()
    answers = [ANSWERS[key][l] for l in LABELS[key]]
    tag = "_".join(args.splits) + (f"_limit{args.limit}" if args.limit else "")

    if args.smoke:
        for p in df["path"].head(2):
            wav = to16(load_audio(p))
            print("free answer:", repr(generate(proc, model, wav, "Describe this music in one sentence.")))
            print("free answer:", repr(generate(proc, model, wav, PROMPTS[key]["plain"])))

    for name in args.prompts:
        prefix = prompt_text(proc, PROMPTS[key][name])
        path = HW1 / "features" / f"{key}_alm_{name}_{tag}.npz"
        done = dict(np.load(path)) if path.exists() else {"logp": np.zeros((0, len(answers))), "sample_id": np.array([], str)}
        logp = list(done["logp"])
        t0 = time.time()
        for i in range(len(logp), len(df)):
            logp.append(score_clip(proc, model, to16(load_audio(df["path"][i])), prefix, answers))
            if (i + 1) % args.save_every == 0 or i + 1 == len(df):
                np.savez(path, logp=np.array(logp), sample_id=df["sample_id"][: i + 1].to_numpy(str),
                         split=df["split"][: i + 1].to_numpy(str), label=df["label"][: i + 1].to_numpy(str),
                         labels=np.array(LABELS[key]), prompt=PROMPTS[key][name])
                print(f"{key} {name} {i + 1}/{len(df)} {(time.time() - t0) / max(1, i + 1 - len(done['logp'])):.2f}s/clip",
                      flush=True)
        lp = np.array(logp)
        y = df["label"].to_numpy(str)
        has = y != ""
        if has.any():
            yi = np.array([LABELS[key].index(l) for l in y[has]])
            o = np.argsort(-lp[has], 1)[:, :3]
            t1, t3 = (o[:, 0] == yi).mean(), (o == yi[:, None]).any(1).mean()
            pred = np.bincount(o[:, 0], minlength=len(answers))
            print(f"{key} {name}: top1 {t1:.3f} top3 {t3:.3f} S {t1 + 0.5 * t3:.3f} | top-1 counts {dict(zip(LABELS[key], pred.tolist()))}")


if __name__ == "__main__":
    main()
