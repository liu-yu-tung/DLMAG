# HW3: text-to-music generation and evaluation

NTU DLMAG HW3. Stable Audio Open 1.0 (`stabilityai/stable-audio-open-1.0`) on the 100 test captions:

- **Retrieval:** top-1 MusicCaps clip per caption with LAION-CLAP (`music_audioset_epoch_15_esc_90.14.pt`) -> `results/retrieval.json`.
- **Generation:** our own classifier-free guidance (`text_to_music/cfg.py`) in a sampling loop that follows the TA sample code (`sample_code/`), plus guidance extensions (CFG rescale, APG, guidance interval, composed guidance) in `text_to_music/sampler.py`.
- **Evaluation:** av-benchmark (FD-VGGish, FD-PANN, FD-PaSST, KL-PANN, KL-PaSST, CLAP 90.14), FD-MERT, MQAScore (Music Flamingo judge), a second judge (Qwen2-Audio), paired per-prompt bootstrap intervals.

All settings were chosen on a dev set of 100 MusicCaps captions (`scripts/retrieval/build_dev.py`, excluding every top-5 test-retrieved clip); the test set was generated with fixed settings. Predictions were written before each result (`docs/predictions.toml`).

## Submitted clips

**Run:** `test/rescale12_best3_c8998`.

- **Guidance:** CFG at scale 12 with CFG rescale, phi 0.7 (Lin et al., WACV 2024), 100 steps, caption prompts.
- **Best of 3:** seeds 0, 1, 2 per caption (`seed = id + 1000 * seed_index`). Per caption, the clip with the highest cosine to its caption under a second CLAP checkpoint (`music_speech_audioset_epoch_15_esc_89.98.pt`) is kept. Scoring uses the 90.14 checkpoint, so the ranker is not the scorer; the CLAP gain of the selection is still partly selection bias (see the report).
- **Picks:** `results/test__rescale12_best3_c8998/selection.csv` (per caption: the three scores and the chosen run; also a release asset). Chosen counts: seed 0 26, seed 1 41, seed 2 33.

| test run, vs cfg7 seed 0 (paired, 95% interval) | CLAP 90.14 | dCLAP | KL-PaSST | dKL | MQAScore |
|---|---|---|---|---|---|
| cfg7 (baseline) | 0.314 | | 0.950 | | 0.651 |
| rescale12, seed 0 (no selection) | 0.325 | +0.010 [-0.000, +0.021] | 0.929 | -0.021 [-0.101, +0.053] | 0.658 |
| rescale12 best of 3 (submitted) | 0.354 | +0.040 [+0.027, +0.051] | 0.883 | -0.067 [-0.170, +0.031] | 0.672 |

Source: `results/analysis/findings_table.md`. Dev results, ablations and the analysis: the report (`b10901114_report.pdf` in the release) and `docs/findings.md`.

## Setup

```sh
uv sync
```

- **Python env:** `pyproject.toml` and `uv.lock` (Python 3.11, torch 2.11.0 from the cu128 index, diffusers 0.41.0). `requirements.txt` holds the same pins for pip.
- **Stable Audio Open weights:** gated on Hugging Face. Accept the license on the model page, then `hf auth login`. Generation needs about 9 GB of GPU memory (fp16; tested on an RTX 4060 Ti 16 GB).
- **CLAP checkpoints:** `music_audioset_epoch_15_esc_90.14.pt` and `music_speech_audioset_epoch_15_esc_89.98.pt` from `lukewys/laion_clap` on Hugging Face, in `external/av-benchmark/weights/`.
- **av-benchmark:** clone with its own venv, upstream commit `f351b9a6fc6abde746d5f8e1d4c47c883319cb41` plus `docs/av_benchmark.patch`; steps in `docs/eval_notes.md` ("Setup").
- **FD-MERT:** `scripts/evaluation/fd_mert.py` imports `hw1.features.mert` from the HW1 project in the same repository (`../hw1`) and runs with its venv; see `docs/eval_notes.md`.
- **MQAScore:** clone of https://github.com/Lonian6/MQAScore at `0ba54014df73fdbffe6c35fa1b42cc679857ab82` plus `docs/mqascore.patch` (bf16 with CPU offload, a Hugging Face tag-extraction backend) and a Music Flamingo venv; steps in `docs/mqa_notes.md`.
- **Data (course material, not in this repo):** `data/test.csv` (100 captions) and `data/musiccaps/audio_mp3/` (5,367 MusicCaps MP3 files).

## Reproduce the submission

```sh
.venv/bin/python scripts/generation/generate.py --no-quick-clap --set test --run rescale12 --scale 12 --rescale-phi 0.7
.venv/bin/python scripts/generation/generate.py --no-quick-clap --set test --run rescale12_seed1 --scale 12 --rescale-phi 0.7 --seed-index 1
.venv/bin/python scripts/generation/generate.py --no-quick-clap --set test --run rescale12_seed2 --scale 12 --rescale-phi 0.7 --seed-index 2
external/av-benchmark/.venv/bin/python scripts/evaluation/evaluate.py --set test --pred outputs/test/rescale12 --name test__rescale12
external/av-benchmark/.venv/bin/python scripts/evaluation/evaluate.py --set test --pred outputs/test/rescale12_seed1 --name test__rescale12_seed1
external/av-benchmark/.venv/bin/python scripts/evaluation/evaluate.py --set test --pred outputs/test/rescale12_seed2 --name test__rescale12_seed2
external/av-benchmark/.venv/bin/python scripts/evaluation/clap_rescore.py --set test --runs rescale12,rescale12_seed1,rescale12_seed2 --ckpt external/av-benchmark/weights/music_speech_audioset_epoch_15_esc_89.98.pt --column clap8998_cosine --device cuda
.venv/bin/python scripts/selection/select_best_of_n.py --set test --runs rescale12,rescale12_seed1,rescale12_seed2 --name rescale12_best3_c8998 --score clap8998_cosine
.venv/bin/python scripts/submission/build_and_check_submission_zip.py --run-dir outputs/test/rescale12_best3_c8998
```

- **Generation** (about 16 s per clip) writes `outputs/test/<run>/<id>.wav` and records in `results/test__<run>/` (`gen_info.json`, `quick.jsonl`, `trajectory.jsonl`); it resumes where it stopped. Defaults: `--mode cfg`, `--steps 100`, `--seconds 10`, `--prompt-style caption`; other flags in `--help`.
- **Evaluation** writes `results/test__<run>/metrics.json` and `per_prompt.csv`; `clap_rescore.py` adds the `clap8998_cosine` column the selection ranks by.
- **Packaging** builds `outputs/submission/b10901114/` (`generated_samples/0.mp3`-`99.mp3`, `cfg.py`, `retrieval.json`) and `b10901114.zip`, and checks it: 100 MP3s decode at 10 s, `retrieval.json` keys and values, `cfg.py` imports, no dotfiles. It also writes a manifest with the sha256 of every file.
- **Exact repeats:** with the same GPU and library versions, regenerated clips are bit-identical (`scripts/generation/check_determinism.py`; checked on 3 baseline clips; not run for rescale12). Other hardware can change the clips slightly, and the selection can flip on near ties (top-2 gaps under 0.01 on 24 of the 100 test captions). To rebuild the submitted zip from regenerated clips, take the picks from the shipped `selection.csv` rather than re-ranking.
- **CPU rescoring:** `clap_rescore.py --device cpu` rounds the model's convolutions to TF32 to match the GPU scores (max difference 1.7e-4 on 10 dev clips); plain fp32 on CPU differs by up to 0.022.

Check our CFG step against diffusers' own pipeline (TA check, max difference 0):

```sh
cd sample_code && ../.venv/bin/python generate.py --prompt 'jazz, electric guitar, piano, relaxing' --out ../logs/check.wav --check
```

## Other commands

- **Retrieval:** `external/av-benchmark/.venv/bin/python scripts/retrieval/retrieval.py` -> `results/retrieval.json` (keys "0"-"99", MusicCaps ids), `retrieval_top5.csv`, `retrieval_ceiling.csv`.
- **FD-MERT:** `../hw1/.venv/bin/python scripts/evaluation/fd_mert.py --pred outputs/test/<run> --name test__<run>` (after `evaluate.py`).
- **MQAScore:** `.venv/bin/python scripts/evaluation/mqa.py --run test__<run> --audio-dir outputs/test/<run> --tags outputs/mqa/tags_test.jsonl`.
- **Second judge:** `../hw1/.venv/bin/python -P scripts/evaluation/alm_judge.py --set dev --run <run> --audio-dir outputs/dev/<run> --tags outputs/mqa/tags_dev.jsonl`.
- **Analysis:** every script in `scripts/analysis/` writes a table or figure cited in `docs/findings.md`, `docs/deck_numbers.md` or `docs/report_narrative.md`; `scripts/analysis/knowledge.py` (prediction verdicts, `results/analysis/knowledge.md`), `findings_table.py`, `deck_numbers.py` (writes `docs/deck_numbers.md`), `vocal_split.py`, `caption_classes.py`; paired FD-VGG: `scripts/evaluation/paired_fd.py`; review checks in `scripts/analysis/review_checks/` (numbers in `docs/analysis_review.md`).
- **Listening tests:** `scripts/listening/listening.py` (`prepare`, `import`, `reveal`); records in `results/listening/`.
- **Experiment queue:** every run in the report is a job in `queue.toml`, run in order by `scripts/generation/job_queue.py`.

## Tests

```sh
.venv/bin/python tests/test_guidance_ext.py
.venv/bin/python tests/test_apg.py
```

`tests/fixtures/sampler_before_guidance_extensions.py` is the sampler before the guidance extensions; the byte-for-byte checks compare against it.

## File map

- `text_to_music/`: the importable package. `cfg.py` is the CFG exercise (the submitted `cfg.py` is this file); `sampler.py` the sampling loop and guidance variants (CFG rescale, APG, interval, composed); `prompts.py` caption and tag prompts; `eval_common.py` shared paths and metric helpers; `quick_clap.py` a running CLAP estimate.
- `sample_code/`: the course `sample_code.zip` as given; `cfg.py` links to ours (see its README).
- `scripts/retrieval/`: top-1 retrieval, its HF cross-check and caption statistics, and the dev-set builder.
- `scripts/generation/`: the generation loop (`generate.py`), the experiment queue runner, caption variants for the word studies (edits, cleaned, degraded, followed words, probe loop), the determinism check.
- `scripts/evaluation/`: av-benchmark metrics, FD-MERT, MQAScore and its validation, second judge, CLAP rescoring, extra metrics, paired FD, FD floor, vocal stems, embeddings of generated clips.
- `scripts/selection/`: best-of-N selection and the probe-based ranker.
- `scripts/submission/`: `build_and_check_submission_zip.py` builds and checks the course zip.
- `scripts/listening/`: listening tests and the non-blind samples page.
- `scripts/analysis/`: tables and figures cited in the docs; `review_checks/` holds the review checks.
- `results/`: run records (`<set>__<run>/`), `analysis/`, `listening/`. No audio is tracked; clips are in the release.
- `docs/`: `findings.md`, `deck_numbers.md`, `report_narrative.md`, `predictions.toml`, `analysis_review.md`, setup notes and patches.
- Not in git, symlinked from `~/Developments/DLMAG/hw3/`: `data/`, `outputs/` (clips, caches), `external/` (av-benchmark, MQAScore), `logs/`, `report/`. `queue.toml` and the notes `docs/status.md`, `docs/handover.md`, `docs/methods_research.txt` are local files, also not tracked.

## References

The full list is in the report (`b10901114_report.pdf` in the release).
