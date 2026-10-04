"""Inference: audio -> features -> checkpoints -> top-3 labels per clip, written as the submission JSON.

    python scripts/predict.py --data /path/to/data --out r12345678.json

--data is a folder holding dataset_A/ and dataset_B/ (each with manifest.csv and audio/). Only rows of
--split (default test) are predicted. Output: {"dataset_A": {sample_id: [top1, top2, top3]}, "dataset_B": ...}.
"""

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# src/ on the path, so the hw1 package works without installing it
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hw1 import final
from hw1.data import load_audio, load_manifest
from hw1.features.langid import LangID
from hw1.features.mert import load_mert, pooled_layers
from hw1.metrics import top3_labels, validate_predictions

HW1 = Path(__file__).resolve().parents[1]


def mert_features(model, paths: list[str], batch: int) -> np.ndarray:
    batches = [paths[i : i + batch] for i in range(0, len(paths), batch)]
    out = []
    with ThreadPoolExecutor(4) as pool:
        for wavs in tqdm(pool.map(lambda b: [load_audio(p) for p in b], batches), total=len(batches), desc="MERT"):
            lengths = [len(w) for w in wavs]
            x = np.zeros((len(wavs), max(lengths)), dtype=np.float32)
            for i, w in enumerate(wavs):
                x[i, : len(w)] = w
            out.append(pooled_layers(model, x, None if len(set(lengths)) == 1 else lengths))
    return final.mert_means(np.concatenate(out))


def ft_features(model, ft: dict, ckpt_dir: Path, n_cls: int, paths: list[str], batch: int, device: str) -> np.ndarray:
    """Fine-tuned top blocks, as in scripts/finetune_mert.py: the frozen block-K output of a full MERT-v2 pass, rounded
    through fp16 as cached for training, then each saved stack (bf16 autocast) gives log-probs; mean over seeds."""
    from finetune_mert import Top

    k = ft["from_layer"]
    hidden = []
    with torch.inference_mode(), ThreadPoolExecutor(4) as pool:
        batches = [paths[i : i + batch] for i in range(0, len(paths), batch)]
        for wavs in tqdm(pool.map(lambda b: [load_audio(p) for p in b], batches), total=len(batches), desc="MERT block states"):
            groups = [wavs] if len({len(w) for w in wavs}) == 1 else [[w] for w in wavs]
            for g in groups:
                x = torch.as_tensor(np.stack(g), device=device)
                hs = model(input_values=x, output_hidden_states=True, return_dict=True).hidden_states
                hidden += list(hs[k - 1].half().cpu())
    same = len({h.shape for h in hidden}) == 1
    chunks = [torch.stack(hidden[i : i + 16]) for i in range(0, len(hidden), 16)] if same else [h[None] for h in hidden]
    runs = []
    for name in ft["checkpoints"]:
        sd = torch.load(ckpt_dir / name, map_location=device)
        top = Top(model, k, n_cls).to(device)
        top.blocks.load_state_dict(sd["blocks"])
        top.head.load_state_dict(sd["head"])
        top.eval()
        out = []
        with torch.inference_mode(), torch.autocast(device.split(":")[0], dtype=torch.bfloat16):
            for h in tqdm(chunks, desc=name):
                out.append(torch.log_softmax(top(h.to(device).float()).float(), -1).cpu().numpy())
        runs.append(np.concatenate(out))
        del top, sd
        torch.cuda.empty_cache()
    return np.mean(runs, axis=0)


def lang_features(lid: LangID, paths: list[str], batch: int = 16) -> np.ndarray:
    batches = [paths[i : i + batch] for i in range(0, len(paths), batch)]
    with ThreadPoolExecutor(4) as pool:
        out = [lid.log_probs(wavs, 24000) for wavs in tqdm(pool.map(lambda b: [load_audio(p) for p in b], batches),
                                                         total=len(batches), desc="Whisper language ID")]
    return np.concatenate(out).astype(np.float32)


def alm_features(key: str, paths: list[str], device: str) -> np.ndarray:
    """Zero-shot Qwen2-Audio label log-likelihoods (plain prompt), as in scripts/alm_qwen.py."""
    from alm_qwen import ANSWERS, PROMPTS, load_model, prompt_text, score_clip, to16

    proc, qwen = load_model(device)
    prefix = prompt_text(proc, PROMPTS[key]["plain"])
    answers = [ANSWERS[key][l] for l in final.LABELS[key]]
    out = np.stack([score_clip(proc, qwen, to16(load_audio(p)), prefix, answers) for p in tqdm(paths, desc="Qwen2-Audio")])
    del qwen
    torch.cuda.empty_cache()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True, help="folder containing dataset_A/ and dataset_B/")
    ap.add_argument("--out", type=Path, required=True, help="output JSON path")
    ap.add_argument("--ckpt-dir", type=Path, default=HW1 / "checkpoints")
    ap.add_argument("--split", default="test")
    ap.add_argument("--datasets", nargs="+", default=["A", "B"])
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-alm", action="store_true",
                    help="skip Qwen2-Audio (default mode peaks at about 13 GB GPU memory) and use each checkpoint's "
                         "fallback recipe")
    args = ap.parse_args()

    model = load_mert(device=args.device)
    preds, manifests, lid = {}, {}, None
    for key in args.datasets:
        df = load_manifest(args.data / f"dataset_{key}")
        manifests[f"dataset_{key}"] = df
        df = df[df["split"] == args.split].reset_index(drop=True)
        if df.empty:
            raise ValueError(f"no rows with split={args.split} in dataset_{key}")
        ckpt = final.load(args.ckpt_dir / f"{key}.joblib")
        paths = df["path"].tolist()
        recipe = ckpt["recipe"]
        if args.no_alm and "alm_qwen" in recipe:
            recipe = ckpt["fallback"]
        mert = mert_features(model, paths, args.batch)
        feats = {"mert_layeravg": mert, "mert_ord10": mert}
        if "mert_ft12" in recipe:
            feats["mert_ft12"] = ft_features(model, ckpt["ft"], args.ckpt_dir, len(ckpt["labels"]), paths, args.batch,
                                             args.device)
        if "alm_qwen" in recipe:
            feats["alm_qwen"] = alm_features(key, paths, args.device)
        if "lang_mixture" in recipe:
            lid = lid or LangID(device=args.device)
            feats["lang_mixture"] = lang_features(lid, paths)
        probs = final.predict_proba(ckpt, feats, recipe)
        preds[f"dataset_{key}"] = dict(zip(df["sample_id"].tolist(), top3_labels(probs, ckpt["labels"])))
        print(f"dataset_{key}: {len(df)} clips, recipe {'+'.join(recipe)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(preds, indent=2) + "\n")
    if args.split == "test":
        errors = validate_predictions(preds, manifests)
        if errors:
            raise SystemExit("invalid predictions:\n" + "\n".join(errors))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
