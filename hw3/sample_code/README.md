# Sample code (HW3)

These files come from `sample_code.zip`, provided by the HW3 course staff for NTU Deep Learning for Music Analysis and Generation, 2026. They are kept here so the CFG check runs as submitted.

Changes from the original:
- `cfg.py` is a symlink to our filled-in exercise, `../text_to_music/cfg.py` (the zip holds the empty stub). The submitted `cfg.py` in the zip is the same file.
- `test.csv` from the zip is left out (course data; the captions live in `../data/test.csv`).
- `generate.py`, `requirements.txt` and the README below are byte-identical to the zip.

Check (from this folder, with the project venv): `../.venv/bin/python generate.py --prompt 'jazz, electric guitar, piano, relaxing' --out ../logs/check.wav --check`

The original README follows.

---

# HW3 sample code: classifier-free guidance for Stable Audio Open

## Setup
```bash
conda create -n deepmir_hw3 python=3.11 && conda activate deepmir_hw3
pip install torch==2.11.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```
`stabilityai/stable-audio-open-1.0` is gated on Hugging Face: accept its license, then run `huggingface-cli login`.
Generation needs about 9 GB of GPU memory (fp16).

## Files
| File | Description |
| --- | --- |
| `cfg.py` | **Exercise**: implement `cfg(noise_pred_uncond, noise_pred_cond, guidance_scale)`. |
| `generate.py` | The `StableAudioPipeline` sampling loop with the CFG step replaced by your `cfg`. |

## Exercise
1. Implement `cfg` in `cfg.py`.
2. Generate audio and compare with diffusers' own pipeline:
```bash
python generate.py --prompt "jazz, electric guitar, piano, relaxing" --out out.wav --check
```
With a correct implementation the difference is `0` (written in the same form as diffusers) or very small.

Note: Stable Audio Open is a v-prediction model, so the model outputs are v rather than noise. CFG is linear, so the
formula is the same.
