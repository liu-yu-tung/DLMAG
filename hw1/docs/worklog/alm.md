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

## Qwen vs Whisper language ID on B (2026-09-29 17:10; train out-of-fold, Qwen = fold-fitted remap of its zero-shot scores, Whisper = logistic probe on its 100 language log-probs)

- **Nearly the same information.** Top-1 0.575 for both; same top-1 prediction on 71% of clips; correctness correlation 0.68; each is right alone on only 7.8% of clips, both wrong on 34.7%.
- **Same per-market profile (top-1):** Brazil 0.88 / 0.88, Italy 0.63 / 0.55, Spain 0.61 / 0.59, US 0.56 / 0.59, UK 0.49 / 0.53, Germany 0.28 / 0.32 (Qwen / Whisper). Both fail where the singing is in English.
- **Qwen's raw answer follows the sung language** (cross-table of Whisper's top-1 language vs Qwen's zero-shot answer): pt -> Brazil 117/119, de -> Germany 22/24, es -> Spain 71/81, en -> US 384/481 (UK only 57). Differences: Italian is often answered as Spain (22/63), and English is almost always mapped to the US.
- **Small complementary part:** Qwen + Whisper (calibrated) 1.026 against Qwen 0.998 (+0.028, CI 0.005 to 0.051); once MERT is in the fusion the Qwen gain shrinks to about +0.01 (not significant).
- Reading: zero-shot Qwen answers the market question mostly by "detect language, map to country, English means US".

## Explanations (2026-09-29 17:23, `scripts/alm_explain.py`, `results/alm_explanations.md`)

- 16 validation clips (per task, 4 ranked right and 4 wrong), plain prompt plus "explain in two or three sentences which sounds led you to that answer", greedy.
- **The reasons cite a "description" that does not exist** ("the description mentions an 'American' sound", "a '50s vibe'"): the model talks as if it had been given a caption, a trace of its caption-style training data.
- **Genre-to-era shortcuts:** "country rock -> 1970s", "bossa nova, acoustic guitar -> 1960s", "synthesizers and electronic beats -> 2010s"; on B, "English lyrics, rock -> United States".
- **Factual and perceptual errors:** "country rock emerged in the mid-1970s"; "'76' is a common British number plate format"; a German clip described as "a female vocalist singing in Spanish"; one answer outside the label set (1950s).
- **Free answers often disagree with the ranking:** 5 of 16 (for example ranked 1970s but answered 1960s; ranked Germany but answered Spain). The explanations are post-hoc stories, not the mechanism behind the scores.

## A train/test scores and screen (2026-09-29 17:50, `results/cv_alm.json`)

- **Alone (out-of-fold / validation):** raw 0.766 / 0.731; label-mean removed 0.810 / 0.780; remap 0.839 / 0.886.
- **The most complementary A component so far:** on MERT's 529 misses, true-class rank 2.70-2.75 and top-1 0.28 (Demucs 0.23, CNN 0.23-0.25).
- **With MERT (calibrated):** + Qwen (label-mean) 0.931 (+0.026, CI -0.001 to 0.052); ordinal MERT (eps 0.1) + Qwen 0.947 (+0.042, CI 0.014 to 0.071), validation 0.977 (current 0.943). This is the first A recipe to clear the rule (>0.03, CI above zero). Caveat: the label-mean correction was picked from two options; eps 0.1 was fixed in advance.
- **Cost if adopted:** `predict.py` for A would need Qwen2-Audio (16 GB download, bitsandbytes, about 10 GB GPU, about 3 s per clip, about 7 min for 132 test clips). Decision pending.
