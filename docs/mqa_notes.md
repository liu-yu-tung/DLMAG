# MQAScore setup notes (HW3)

MQAScore by the course TA: https://github.com/Lonian6/MQAScore, commit `0ba54014df73fdbffe6c35fa1b42cc679857ab82`, cloned to `HW3/external/MQAScore` (gitignored, so the patch below is the record of our changes).

## Install

One uv venv serves both Music Flamingo scoring and tag extraction (the repo's two conda envs differ only in vLLM / Qwen3-Omni, which we do not use).

Pins follow `envs/music_flamingo_requirements.txt` (python 3.12, torch 2.11.0, transformers 5.8.0.dev0 at git commit `41c3a5ac`, accelerate 1.13.0, librosa 0.11.0, numpy 2.4.4, ...). Dropped: the CUDA 13 `nvidia-*` pins and `triton` (torch cu128 brings its own), `flash_attn` (needs a compile; scoring uses the default attention, so the numbers can differ slightly from the paper env), and notebook/plot packages.

```
cd HW3/external/MQAScore
uv venv --python 3.12 .venv-mf
uv pip install --python .venv-mf/bin/python --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match torch==2.11.0 -r <pins with the drops above>
```

Weights (about 16.5 GB and 8.1 GB) are in the normal Hugging Face cache: `nvidia/music-flamingo-hf`, `Qwen/Qwen3-4B-Instruct-2507`.

## Offload patch (bf16 kept, no quantization)

Music Flamingo needs about 16.5 GB in bf16, so it does not fit in 16 GB. `--gpu-memory-gib N` loads it with `device_map="auto"` and `max_memory={0: "N GiB", "cpu": ...}`; layers that do not fit stay in CPU RAM and accelerate moves them to the GPU for each forward pass. Batch size stays 1. Math is unchanged, only weight placement differs. The default (flag absent) behaviour is the original.

Tag extraction: the repo's `extract_tags.py` needs vLLM (another torch, about 10 GB of wheels, no room on this disk). `--backend hf` runs the same instruction with transformers `generate`, greedy, one prompt at a time. Default is still `vllm`.

Diff against the cloned commit:

```diff
diff --git a/mqascore/models/audio_flamingo.py b/mqascore/models/audio_flamingo.py
index 95f2a2c..e0a2cd8 100644
--- a/mqascore/models/audio_flamingo.py
+++ b/mqascore/models/audio_flamingo.py
@@ -21,7 +21,8 @@ VARIANTS = {
 class AudioFlamingoBackend(LALMBackend):
     default_fix_seconds = 30
 
-    def __init__(self, variant="af", model_id=None, device="cuda"):
+    def __init__(self, variant="af", model_id=None, device="cuda",
+                 gpu_memory_gib=None, cpu_memory_gib=None):
         import transformers
         from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor
 
@@ -31,17 +32,20 @@ class AudioFlamingoBackend(LALMBackend):
         self.name = variant
         self.model_id = model_id or default_id
         self.device = device
+        load_kwargs = {"device_map": device, "torch_dtype": torch.bfloat16}
+        if gpu_memory_gib is not None:
+            load_kwargs["device_map"] = "auto"
+            load_kwargs["max_memory"] = {0: f"{gpu_memory_gib}GiB", "cpu": f"{cpu_memory_gib or 40}GiB"}
         try:
             self.model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
-                self.model_id, device_map=device, torch_dtype=torch.bfloat16).eval()
+                self.model_id, **load_kwargs).eval()
         except (ValueError, OSError, TypeError) as e:
             cls = getattr(transformers, fallback, None) if fallback else None
             if cls is None:
                 raise
             print(f"[{variant}] AudioFlamingo3ForConditionalGeneration failed ({e}); "
                   f"trying {fallback}", flush=True)
-            self.model = cls.from_pretrained(
-                self.model_id, device_map=device, torch_dtype=torch.bfloat16).eval()
+            self.model = cls.from_pretrained(self.model_id, **load_kwargs).eval()
         self.processor = AutoProcessor.from_pretrained(self.model_id)
         self.yes_ids, self.no_ids = yes_no_ids(self.processor.tokenizer)
         self.yes_t = torch.tensor(self.yes_ids, device=device)
@@ -69,5 +73,5 @@ class AudioFlamingoBackend(LALMBackend):
         am = inputs["attention_mask"]
         # Position of the last real token; correct for left, right or no padding.
         last = am.shape[1] - 1 - am.flip(1).argmax(dim=1)
-        nl = logits[torch.arange(logits.shape[0], device=logits.device), last]
-        return pyes_from_logits(nl, self.yes_t, self.no_t)
+        nl = logits[torch.arange(logits.shape[0], device=logits.device), last.to(logits.device)]
+        return pyes_from_logits(nl, self.yes_t.to(nl.device), self.no_t.to(nl.device))
diff --git a/scripts/extract_tags.py b/scripts/extract_tags.py
index cc22544..311b091 100644
--- a/scripts/extract_tags.py
+++ b/scripts/extract_tags.py
@@ -50,6 +50,24 @@ def extract_json(text):
     return {k: obj.get(k, []) for k in ALL_ATTRIBUTES}
 
 
+def generate_hf(model_id, convs, max_new_tokens, temperature):
+    import torch
+    from tqdm import tqdm
+    from transformers import AutoModelForCausalLM, AutoTokenizer
+
+    tokenizer = AutoTokenizer.from_pretrained(model_id)
+    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="cuda").eval()
+    sampling = {"do_sample": True, "temperature": temperature} if temperature > 0 else {"do_sample": False}
+    texts = []
+    for conv in tqdm(convs):
+        ids = tokenizer.apply_chat_template(conv, add_generation_prompt=True, return_tensors="pt",
+                                            return_dict=True).to("cuda")
+        with torch.no_grad():
+            out = model.generate(**ids, max_new_tokens=max_new_tokens, **sampling)
+        texts.append(tokenizer.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True))
+    return texts
+
+
 def parse_args():
     p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
     p.add_argument("--input", required=True)
@@ -60,6 +78,8 @@ def parse_args():
     p.add_argument("--max-model-len", type=int, default=8192)
     p.add_argument("--temperature", type=float, default=0.0, help="0.0 = greedy")
     p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
+    p.add_argument("--backend", choices=["vllm", "hf"], default="vllm",
+                   help="hf = transformers generate, one prompt at a time (no vLLM install needed)")
     return p.parse_args()
 
 
@@ -76,21 +96,23 @@ def main():
     if not todo:
         return
 
-    from vllm import LLM, SamplingParams
-
-    llm = LLM(model=args.model, max_model_len=args.max_model_len,
-              gpu_memory_utilization=args.gpu_memory_utilization)
-    params = SamplingParams(temperature=args.temperature, max_tokens=args.max_new_tokens)
     instruction = tag_extraction_instruction()
     convs = [[{"role": "system", "content": instruction},
               {"role": "user", "content": it["text"]}] for it in todo]
-    outputs = llm.chat(convs, params)
+    if args.backend == "vllm":
+        from vllm import LLM, SamplingParams
+
+        llm = LLM(model=args.model, max_model_len=args.max_model_len,
+                  gpu_memory_utilization=args.gpu_memory_utilization)
+        params = SamplingParams(temperature=args.temperature, max_tokens=args.max_new_tokens)
+        raw_outputs = [o.outputs[0].text for o in llm.chat(convs, params)]
+    else:
+        raw_outputs = generate_hf(args.model, convs, args.max_new_tokens, args.temperature)
 
     if fail_path.exists():
         fail_path.unlink()
     n_fail = 0
-    for it, o in zip(todo, outputs):
-        raw = o.outputs[0].text
+    for it, raw in zip(todo, raw_outputs):
         try:
             append_jsonl(out, {**it, "tags": extract_json(raw)})
         except (ValueError, json.JSONDecodeError) as e:
diff --git a/scripts/score.py b/scripts/score.py
index d510966..6663892 100644
--- a/scripts/score.py
+++ b/scripts/score.py
@@ -39,6 +39,10 @@ def parse_args():
     p.add_argument("--cache-dir", default=".mqascore_audio_cache")
     p.add_argument("--model-id", default=None, help="override the HF model id / local path")
     p.add_argument("--attn-impl", default="auto", help="qwen3omni only: auto|flash_attention_2|sdpa")
+    p.add_argument("--gpu-memory-gib", type=float, default=None,
+                   help="af/mf only: cap GPU weights at this many GiB and offload the rest to CPU (bf16 kept)")
+    p.add_argument("--cpu-memory-gib", type=float, default=None,
+                   help="af/mf only, with --gpu-memory-gib: CPU RAM budget for offloaded weights (default 40)")
     p.add_argument("--limit", type=int, default=None, help="score only the first N items (dry run)")
     return p.parse_args()
 
@@ -68,6 +72,8 @@ def main():
     kw = {"model_id": args.model_id} if args.model_id else {}
     if args.model == "qwen3omni":
         kw["attn_impl"] = args.attn_impl
+    elif args.gpu_memory_gib is not None:
+        kw.update(gpu_memory_gib=args.gpu_memory_gib, cpu_memory_gib=args.cpu_memory_gib)
     backend = load_backend(args.model, **kw)
     fs = {"default": {}, "full": {"fix_seconds": None}}.get(
         args.fix_seconds, {"fix_seconds": float(args.fix_seconds)})
```

## Tags for the 100 test captions

`data/test.csv` -> `outputs/mqa/captions_test.jsonl` (`id`, `text`) -> `external/MQAScore/scripts/extract_tags.py --backend hf` (Qwen3-4B-Instruct-2507, bf16, greedy, repo default instruction, max 1024 new tokens). 110 of 110 captions parsed (100 test + 12 MusicCaps validation captions, extracted in one run; 2 of the 112 lines came from a first 2-item trial with the same settings), 0 failures, about 7.5 s per caption.

- `outputs/mqa/tags_test.jsonl`: id, text, tags in the eight attributes (genre/instrument/vocal items are `{entity, ...}` dicts, other attributes plain strings). Attribute order is the repo's `ALL_ATTRIBUTES`.
- `results/tags_test_summary.csv`: per item `n_tags_scored` (the five default attributes: genre, instrument, mood_theme, vocal, context, deduplicated as score.py does), `n_tags_all` (all eight), and the entities per attribute joined with "; ".
- Result: 575 scored tags over 100 items (mean 5.75, min 2, max 13, none empty), 734 over all eight attributes.

## Scoring a run

```
uv run python scripts/evaluation/mqa.py --run <name> --audio-dir <dir with 0.wav .. 99.wav> --tags outputs/mqa/tags_test.jsonl
```

- Needs about 15.5 GiB of free GPU memory (`--gpu-memory-gib 13` plus activations); the wrapper refuses to start otherwise. `--allow-busy-gpu` skips the check.
- Runs `scripts/score.py --model mf --protocol pertag` in `external/MQAScore/.venv-mf` with the default attributes and the model default audio length (30 s, zero-padded; the 10 s clips are padded).
- Intermediate files: `outputs/mqa/<name>/` (`items.jsonl`, `score.jsonl`, `score.jsonl.meta.json`, `wall_seconds.txt`). Re-running the same command resumes; `--summarize-only` rebuilds the summary from `score.jsonl`.
- Results: `results/<name>/mqa_summary.json` (mean MQAScore over items, bootstrap 95% CI over items, 1000 resamples, seed 0; per-attribute means over the items that have that attribute, with item counts) and `results/<name>/mqa_per_item.csv`.
- CPU-side checks done: wrapper summary/CSV path on fake scores, processor and chat template on one clip (801 input tokens per question for a 30 s clip). The GPU path is not yet run end to end.
- Gotcha: `scripts/queue.py` shadows the stdlib `queue` module when a script in `scripts/` is run with `python` and imports torch; run `scripts/evaluation/mqa_validate.py` with `python -P`.

## Validation (pending GPU)

The repo ships no reference scores (`tests/fixtures/` only has a README), so `compare_reference.py` has nothing to compare against. Plan instead, on 12 MusicCaps clips with their own caption tags (`outputs/mqa/items_validation.jsonl`, 79 questions or so):

1. Determinism: the same items twice at `--gpu-memory-gib 13` -> `compare_reference.py --tol 0`.
2. Offload-split invariance: `--gpu-memory-gib 13` vs `--gpu-memory-gib 10` -> max and mean |dP(Yes)|.
3. Sanity: matched tags vs the same clips with another clip's tags (`items_validation_mismatched.jsonl`); matched should score clearly higher.
4. Seconds per question from `scripts/evaluation/mqa_validate.py`.

```
cd HW3
V=external/MQAScore/.venv-mf/bin/python
O=outputs/mqa
$V -P scripts/evaluation/mqa_validate.py --gpu-memory-gib 13 --cache-dir $O/audio_cache --jobs $O/items_validation.jsonl:$O/val_a1.jsonl $O/items_validation_mismatched.jsonl:$O/val_mismatch.jsonl $O/items_validation.jsonl:$O/val_a2.jsonl
$V -P scripts/evaluation/mqa_validate.py --gpu-memory-gib 10 --cache-dir $O/audio_cache --jobs $O/items_validation.jsonl:$O/val_b1.jsonl
$V external/MQAScore/tests/compare_reference.py --ours $O/val_a2.jsonl --reference $O/val_a1.jsonl --tol 0
$V external/MQAScore/tests/compare_reference.py --ours $O/val_b1.jsonl --reference $O/val_a1.jsonl --tol 1e-2
```
