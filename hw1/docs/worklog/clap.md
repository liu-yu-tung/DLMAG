# CLAP embeddings and zero-shot — worklog

Status: paused, waiting for `laion/larger_clap_music` weights to finish downloading in the
shared cache. Nothing below past "Changes" has been run end-to-end yet.

## Changes

- `src/hw1/features/clap.py`: reusable CLAP functions.
  - `load_model(device)` -> `(ClapModel, ClapProcessor)` via `from_pretrained("laion/larger_clap_music")`.
  - `resample_to_48k(waveform, src_sr=24000)`: `torchaudio.functional.resample` 24k -> 48k.
  - `split_windows(waveform_48k, n_windows=3, window_s=10)`: splits a 48k waveform into three
    non-overlapping 10 s windows (480000 samples each); zero-pads if the input is short.
  - `embed_waveform_batch(model, processor, waveforms, device, sampling_rate=48000)`: the
    reusable "waveform batch in, embeddings out" function — calls
    `processor(audios=waveforms, sampling_rate=48000, return_tensors="pt")` then
    `model.get_audio_features(**inputs)`. Returns raw (unnormalized) `(N, D)`.
  - `embed_clips(model, processor, waveforms_24k, device, batch_size=16)`: full per-clip
    pipeline — resample, split into 3 windows, batch through `embed_waveform_batch`, mean the
    3 window embeddings per clip, L2-normalize. Returns `(N_clips, D)` float32.
  - `embed_text(...)`, `class_text_embeddings(...)`: text side, same L2-normalize-after-average
    pattern for prompt ensembling per class.
  - `build_prompts_A/B`, `build_prompts(key)`: P1/P2 prompt sets (see Decisions).
  - `cosine_scores(audio_emb, text_emb)`: `audio_emb @ text_emb.T` for two L2-normalized inputs.
- `scripts/extract_clap.py`: thin CLI. `--dataset {A,B,<path>}`, `--out` (default `features`),
  `--batch-size` (default 16). Loads manifest via `hw1.data.load_manifest`/`LABELS`, embeds all
  rows, computes P1/P2 zero-shot cosine scores, saves
  `<out>/<key>_clap.npz` with `emb`, `zs_p1`, `zs_p2`, `sample_id`, `split`, `label`, `labels`,
  `prompts_p1`, `prompts_p2` (json strings), and prints elapsed time and peak GPU memory.
- Used `hw1.data.load_manifest`, `hw1.data.load_audio`, `hw1.data.LABELS` from the shared
  `src/hw1/data.py` module rather than reading the manifest directly — it was already present
  when I started.
- `src/hw1/features/__init__.py` already existed (empty), left as is.

## Decisions

- **Processor settings** — read directly from the already-downloaded (small, complete)
  `preprocessor_config.json` blob in the HF cache for this exact checkpoint (not just library
  defaults), since the full weights aren't in yet:
  - `sampling_rate`: 48000
  - `feature_size` (mel bins): 64
  - `max_length_s` / `chunk_length_s`: 10, i.e. `nb_max_samples`: 480000
  - `truncation`: `"rand_trunc"` — **not** the `ClapFeatureExtractor` class default of
    `"fusion"**; this checkpoint's own config overrides it. Noted because it matters: had we
    fed windows longer than 10 s, the processor would randomly truncate rather than fuse. Since
    we pre-split into exact 480000-sample (10 s @ 48k) windows, input length equals
    `nb_max_samples` exactly and neither truncation nor padding should trigger.
  - `padding`: `"repeatpad"` (also should not trigger for our exact-length windows).
  - Tokenizer: `RobertaTokenizer`, `model_max_length`: 512 (all our text prompts are short,
    well under this).
  - Model config: `projection_dim`: 512, so audio/text embeddings should be dim 512 (**not yet
    verified against a live forward pass**).
- **Window strategy**: 30 s clip -> resample 24k->48k -> split into 3 non-overlapping 10 s
  windows (0-10s, 10-20s, 20-30s) -> embed each -> mean the 3 embeddings -> L2-normalize. This
  exactly matches the checkpoint's `max_length_s=10`/`nb_max_samples=480000`, so no
  truncation/padding logic should be exercised in practice (per config above, untested).
- **Prompts**:
  - Dataset A, P1 (simple, 1 template/class): `"music from the {decade}"`.
  - Dataset A, P2 (descriptive, 2 templates/class, averaged then L2-normalized):
    `"a {decade} song with typical {decade} production"` and
    `"a song produced in the {decade} with {decade}-style production and mixing"`.
  - Dataset B, P1 (simple, 1 template/class): `"music released in {country} in the 1980s"`,
    with natural names (the United States, the United Kingdom, Brazil, Spain, Germany, Italy).
  - Dataset B, P2 (descriptive, 2 templates/class, averaged then L2-normalized):
    `"a 1980s {adjective} pop song"` and `"a {adjective} pop song from the 1980s"`, adjectives
    American/British/Brazilian/Spanish/German/Italian.
  - Ensembling = embed each template separately, mean the raw embeddings within a class, then
    L2-normalize the mean (matches how the audio-side window average is normalized).

## Tests / checks

- `uv run python -c "from hw1.data import load_manifest, LABELS; ..."` — **ran successfully**.
  Dataset A manifest: 1290 rows (train 1026 / validation 132 / test 132). `LABELS` dict matches
  the assignment's class order for A and B.
- Inspected `ClapFeatureExtractor.__init__` and `ClapProcessor` source in the installed
  `transformers==5.17.0` to confirm the API shape (`processor(audios=..., sampling_rate=...)`
  for audio, `processor(text=..., padding=True)` for text, `model.get_audio_features` /
  `model.get_text_features`) — **not yet confirmed against a live model instance**.
- Attempted to load `ClapModel`/`ClapProcessor` via `from_pretrained` twice (once via a
  detached `nohup`, once via `run_in_background`); both were still downloading (network here is
  slow, ~1 MB/s, unauthenticated HF) when the coordinator asked to pause. I killed my own
  in-flight process (PIDs 127032/127034/127037) as instructed and left the shared download
  queue (PID 127073, `HF_HUB_DISABLE_XET=1`, MERT first then CLAP) untouched. No polling since.
- **Not yet run**: `scripts/extract_clap.py` end-to-end on A or B, the zero-shot
  top-1/top-3 accuracy, the linear-probe (StandardScaler + LogisticRegression) eval, or any
  runtime/peak-GPU-memory measurement. `results/clap_eval.json` does not exist yet.
- Confirmed no stray CLAP-loading python processes remain after the kill
  (`ps aux | grep clap` clean except the coordinator's shared download-queue shell).

## Open issues

- CLAP weights (`model.safetensors`/`pytorch_model.bin`, ~740 MB) not yet in the HF cache;
  only small config/tokenizer files and a partial `.incomplete` blob are present. Everything
  downstream of `load_model()` is blocked on this.
- Embedding dimension (expected 512 from `projection_dim`), actual tokenizer/audio-input
  tensor shapes from `processor(...)`, and whether `truncation="rand_trunc"`/`padding` ever
  trigger for our exact 480000-sample windows are all inferred from config, not yet verified
  by running the model.
- GPU is shared with another ~632M-parameter worker; batch size 16 in `extract_clap.py` is a
  guess, not yet checked against actual memory headroom.
- `results/` and `features/` output directories exist but are empty; `results/clap_eval.json`
  and `features/A_clap.npz` / `features/B_clap.npz` are not yet produced.
