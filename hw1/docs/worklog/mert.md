# MERT feature extraction worklog

Status 2026-09-28 23:00: paused while the weights download. Code is written. Only the architecture is tested so far, with random weights. Nothing has run on the real checkpoint yet.

## Changes

- `src/hw1/features/__init__.py`: new, empty.
- `src/hw1/features/mert.py`: new.
  - `load_mert(model_id, dtype, device)`: `AutoModel.from_pretrained(..., trust_remote_code=True, dtype=..., attn_implementation="sdpa")`, then eval and move to device. `dtype` is one of `fp32`, `bf16`, `fp16`.
  - `pooled_layers(model, wavs, lengths=None) -> np.ndarray`: input is `(B, T)` float waveforms at 24 kHz. Output is `(B, L, 2*D)` float32: for each layer, the mean over valid frames concatenated with the std. It uses `feature_attention_mask`. `lengths` builds a right-padded waveform mask for clips of different lengths. This is the function `predict.py` should reuse.
- `scripts/extract_mert.py`: new CLI.
  - **Args:** `--dataset A|B|<dir> --out --model --dtype --batch 8 --tag`.
  - Reads the manifest with `hw1.data.load_manifest` and loads audio with `hw1.data.load_audio`, 4 loader threads.
  - Processes all rows and writes `features/{A,B}_{mertv2|mertv1}.npz`.
  - **npz contents:** `feats` float16 `(N, L, 2D)`, plus `sample_id`, `split`, `label` as str arrays, and `meta` as a JSON string.
  - Writes a `.json` copy of the meta next to the npz: model, dtype, transformers and torch versions, num_layers, hidden_size, seconds, peak_gpu_mib.
- `scripts/smoke_mert.py`: runs one dataset A clip in fp32, bf16 and fp16. It prints the hidden-state count and shapes and the peak GPU memory, and compares the pooled bf16/fp16 output to fp32 (cosine similarity, relative error).
- `scripts/sanity_mert.py [tag]`: checks shape and finiteness for A and B. Then it fits StandardScaler + LogisticRegression(max_iter=2000) on train and scores validation top-1 and top-3 on layers 6, 12, 23 (0-based block index, mean part only). Results go to `results/mert_sanity.json`.
- No changes to `pyproject.toml` or `uv.lock`. No commits.

## Decisions

- **No transformers downgrade so far.** `modeling_mert2.py` builds and runs a forward pass on transformers 5.17.0 with torch 2.14.0+cu130. The only output is a deprecation warning: "`use_return_dict` is deprecated! Use `return_dict` instead!" Our code passes `return_dict=True`. Loading real weights on 5.17 is still untested; `dtype=` in `from_pretrained` is the v5 name.
- **No feature extractor.** `preprocessor_config.json` is a `Wav2Vec2FeatureExtractor` with `do_normalize: false`, so it only pads. We pass the raw float32 waveform. All clips are 720000 samples, so no mask is needed. `lengths` covers the case of different lengths.
- **Precision:** the model's mel frontend always runs in fp32, whatever dtype the model is loaded in (it overrides `_apply`). Pooling runs in fp32 and features are saved as float16. The default dtype for extraction is fp32 until the smoke test compares bf16/fp16 against it.
- **Batch 8:** a 30 s clip gives 750 frames, so attention is small. Stacking 24 layers costs about 73 MB of fp32 per clip. Batch size not yet tuned against the shared GPU (CLAP worker uses about 2-3 GB).

## Tests/checks

- `uv run python -c "import hw1.features.mert"`: imports OK. The editable install picks up the new subpackage.
- Random-init forward on 5.17: `AutoModel.from_config(cfg, trust_remote_code=True)` with one 720000-sample input returns 24 hidden states, each `(1, 750, 1024)`, and a mask of shape `(1, 750)`. `cfg._attn_implementation` was `None` before model init; the model accepted it. We pass `sdpa` explicitly anyway.
- Pending:
  - `uv run python scripts/smoke_mert.py`
  - `uv run python scripts/extract_mert.py --dataset A` (then `B`)
  - `uv run python scripts/sanity_mert.py`

## Open issues

- MERT-v2-30s `model.safetensors` (2529812848 bytes) is not in the cache yet: the download is at about 1 MB/s. Next steps once it lands:
  1. Run the smoke test and record the fp32/bf16/fp16 results.
  2. Pick the dtype.
  3. Extract A and B.
  4. Run the sanity probe.
  5. Update this file.
- If loading real weights fails on 5.17, fall back in this order: a local shim, then an isolated `uv run --isolated --with transformers==4.53.2` check, then MERT-v1-330M (its cache also has no weights yet).

## Results (main session, 2026-09-29)

- **Smoke test on real weights passes on transformers 5.17.0:** 24 layers of (1, 750, 1024), all finite. fp32 takes 0.12 s per clip with a 2.8 GB peak. bf16/fp16 differ from fp32 by up to 32% on some layer, so extraction uses fp32 (see DECISIONS.md).
- **Extraction:** `features/{A,B}_mertv2.npz`, (N, 24, 2048) = per-layer time mean and std. B took 127 s with a 4.7 GB peak.
- **`scripts/sanity_mert.py` removed.** `scripts/bench_mert.py` replaces it: a sweep over all 24 layers with the shared `hw1.probe` logreg, confusion matrices, and `results/mert_layer_sweep.png`.
- **Validation S, logreg on the time-mean of one layer** (chance 0.417; hand-crafted best A 0.792, B 0.681):

| Dataset | Best layer | Top-1 | Top-3 | S | Best layer mean+std | Layer-average |
|---|---:|---:|---:|---:|---:|---:|
| A | 9 | 0.515 | 0.864 | 0.947 | 0.898 | 0.879 |
| B | 14 | 0.578 | 0.843 | 1.000 | 0.966 | 0.966 |

- **The sweep is noisy:** S jumps by up to ±0.1 between neighboring layers on A. The single best layer is picked on 132/102 validation clips, so its S is optimistic. Prefer a band of layers:
  - A: layers 7-12 and 18-24 are all about 0.85-0.92;
  - B: layers 9-24 are all about 0.92-1.00; B's early layers (1-5) are at the hand-crafted level of about 0.67.
- **Tests:** `tests/test_mert_pooling.py` checks the pooling math and padding mask with a fake model (no GPU). 27 tests pass.

## Rerun after the rotary fix (2026-09-29, 01:30)

The numbers above were measured with uninitialized rotary frequencies (see DECISIONS.md, the `restore_rotary` entry) and are superseded. Features were re-extracted (A 160 s), `bench_mert.py` rerun.

| Dataset | Best layer | Top-1 | Top-3 | S | Best layer mean+std | Layer-average (time-mean features) |
|---|---:|---:|---:|---:|---:|---:|
| A | 9 | 0.492 | 0.856 | 0.920 | 0.856 | 0.898 |
| B | 21 | 0.598 | 0.882 | 1.039 | 1.049 | 0.980 |

- **B now has a clear layer structure:** layers 1-5 are at 0.64-0.67 (hand-crafted level), layers 6-11 climb from 0.80 to 0.94, and layers 12-23 sit at 0.95-1.04.
- **A stays flat:** 0.81-0.92 across all layers, with no band clearly better. Era cues are spread through the network.
- **Dtype check rerun (`smoke_mert.py`):** bf16 is now within 0.7% relative error of fp32 (min cosine 0.99997), fp16 within 0.1%, both 3x faster (0.04 s vs 0.12 s per clip) at half the memory (1.5 vs 2.8 GB). The earlier 32% gap was the rotary bug, not precision. The submitted features stay fp32; fp16 is fine for stem features and fine-tuning.
- **Before/after comparison for the report:** old features are in `features/stale_rope/`.
