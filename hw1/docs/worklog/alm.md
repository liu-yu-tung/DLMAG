# Worklog: Qwen2-Audio ALM (required item)

## Setup (2026-09-29, `scripts/alm_qwen.py`)

- `Qwen/Qwen2-Audio-7B-Instruct`, LLM in 8-bit (bitsandbytes), audio tower, projector and lm_head in fp16. Zero-shot: for each clip, log p(label text + `<|im_end|>` | audio, question) for all 6 labels; top-3 = three highest. B labels are spelled out ("United States", "United Kingdom").
- Two prompts per task: plain question, and cue-guided (names what to listen for: production and instruments for A; lyric language, accent and genre for B).
- Download: huggingface-hub 1.33 gives every retry a new temp file name, so interrupted shards restarted from zero; `scripts/resume_qwen.sh` finished them with `curl -C -`, SHA-256 checked against the hub.
- Batch-6 scoring ran out of memory next to the CNN job (full-vocabulary logits); answers are scored 2 per pass, 8.3 s per clip.

## Validation (zero-shot, no training)

| Task | Prompt | Top-1 | Top-3 | S | Top-1 counts |
|---|---|---:|---:|---:|---|
| B | plain | 0.520 | 0.765 | 0.902 | US 50, UK 10, Brazil 20, Spain 18, Germany 2, Italy 2 |
| B | cues | 0.441 | 0.696 | 0.789 | Spain 58, Brazil 19, US 17 |
| A | plain | 0.356 | 0.750 | 0.731 | 1970s 48, 1960s 32, 2010s 31, 1990s 2, 2000s 4 |
| A | cues | 0.356 | 0.720 | 0.716 | 1970s 76, 2010s 26 |

- Chance S 0.417. Trained MERT probes: A 0.943, B 1.034 (B recipe with language 1.069).
- **B is strong for zero-shot** (0.902): the model hears the language. It over-predicts US and almost never says Germany or Italy.
- **A is weaker** (0.731) and skewed: it almost never answers 1990s or 2000s.
- **The cue-guided prompt is worse on both** and makes the skew stronger (Spain on B, 1970s on A).
- **Label bias is large:** removing each label's mean log-prob (fitted on validation itself, so optimistic) gives B 0.98 and A 0.78. A fair version fits this on train; train and test are being scored with the plain prompt.

## B train/test scores and cross-validated screen (2026-09-29 16:45, `scripts/cv_alm.py`, `results/cv_alm.json`)

- Raw zero-shot scores are already out-of-fold (no training). Bias corrections are fitted inside the same 5 train folds.
- **Alone (out-of-fold / validation S):** raw 0.892 / 0.902; label-mean removed 0.956 / 0.985; logistic remap of the 6 scores 0.998 / 1.000. With a train-fitted remap, zero-shot Qwen matches the trained MERT probe (0.964).
- **Added to the B recipe** (MERT + language, calibrated, 1.033 out-of-fold): 1.046 (label-mean), 1.043 (remap); CIs -0.009 to 0.036 and -0.011 to 0.030. Not adopted: under the bar, and it would put a 7B model into `predict.py`.
- **On MERT's 363 misses:** true-class rank 2.63-2.78, top-1 0.32, the same complementarity as Whisper language ID (2.69, 0.33). Qwen's B signal is largely the language signal.
