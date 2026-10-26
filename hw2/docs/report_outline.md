# Report outline (draft 2026-10-25)

PPT-style, 16:9, about 12 pages plus an appendix (spec: about 10 pages, not strict; must read without oral explanation). Every number below is in `results/findings.md` or the CSV named; numbers marked T6 wait for the T6 grades.

## Spec checklist

| Spec item | Slide |
|---|---|
| Cover: name, student ID, cloud-drive link | 1 |
| TASK A: token representation, model architecture | 3, 4 |
| TASK A: implementation details (augmentation, hyper-parameters) | 4 |
| TASK A: at least 3 inference configurations, 20 pieces each, H1/H4/GS (SI optional) next to real data | 5 |
| TASK A: the submitted configuration (20 MIDI + 20 WAV) | 5, 6 |
| TASK B: 3 configurations per prompt, model and settings named, best per prompt | 7 |
| BONUS: model, representation, configuration, CA and OA, ablation | 10 |
| Public code cited (TA tokenizer, `eval_basic.py` / MusDr, `eval_bonus.py`) | 12 |

## Slides

1. **Cover.** Title, name, student ID, cloud-drive link (code, README, checkpoints, generation results).
2. **Overview.** Three tasks, the submitted systems in one table (TASK A single-stage `loss_0.40`, `t1 p1`, first 8 of 40 bars dropped; TASK B same checkpoint, `t1 p0.95`; BONUS model), plus the extra two-stage plan-fill system; how choices were made (blind A/B listening with rules fixed before rating, objective metrics second).
3. **Data and tokens.** POP909: 896 course songs, 45 held out (validation and the real reference), 851 for training. REMI-style tokens from the TA tokenizer, 16 positions per beat, all tracks merged for TASK A/B. Transposition by up to ±5 semitones as augmentation. Plan-fill tokens: per-bar plan labels (key, chords, density and duration bins, repeat labels) and per-bar melody and piano blocks (`docs/plan_fill.md`). Figure: `results/figures/data_distributions.png`.
4. **Model and training.** GPT-2 decoder, 12 layers, 8 heads, 512 wide, 2048 context, 39.2M parameters; AdamW lr 3e-4, warm-up 1000, batch 8, weight decay 0.01, dropout 0.1, stop at train loss 0.08 or 60k steps. Checkpoints saved by training loss (`loss_0.40`) and step; held-out loss rises after about step 7000 while listening still preferred the lower training loss (T1: `loss_0.40` 3 wins, 1 loss, 1 same vs `step_10000`). Table: `results/task_a_checkpoints.csv`.
5. **TASK A configurations and objective metrics.** Table (`results/cand_a_metrics.csv`, `results/baselines.csv`), `loss_0.40`, 20 pieces each:

   | Configuration | H1 | H4 | GS |
   |---|---|---|---|
   | `t1 p1` | 2.180 | 2.621 | 0.970 |
   | `t1 p1`, cut (submitted) | 2.203 | 2.599 | 0.975 |
   | `t1 p0.95` / cut | 2.119 / 2.128 | 2.587 / 2.592 | 0.971 / 0.977 |
   | `t0.9 p1` / cut | 2.116 / 2.107 | 2.581 / 2.567 | 0.968 / 0.973 |
   | real, 32-bar crops | 2.160 | 2.642 | 0.960 |

   Plus the sampling sweep (`results/task_a_sampling.csv`) and the checkpoint rows. Point: the metrics barely separate configurations that listening separates; all are near real.
6. **TASK A choices by listening.** The blind A/B page (pref −2..+2, section duel per 8 bars, flaws); T5: cut beat uncut 4 of 4, and the per-section duel puts the gain in bars 1-8 (uncut lost bars 1-8 in 4 of 5). Sampling fixes: no position past the meter, one opening tempo. Real openings won every anchor pair (T1, T2: 4 of 4).
7. **TASK B.** 3 prompts x 3 configurations (`t1 p1`, `t1 p0.95`, `t0.9 p1`), 8 + 24 bars, same checkpoint as TASK A. Best for every prompt: `t1 p0.95` (T3: 6 of 6 pairs, sign test p 0.031). Example of the failure it avoids: `t0.9 p1` on prompt 3 moves to C# in bars 29-32.
8. **Extra system: plan-fill.** Diagram: plan model writes bar labels (sections, chords, density, repeats; form `I4A8B8A8E4`), fill model writes melody then piano per bar; repeat guide; chord-relative pitches (`Rel_k`); semitone penalty on the piano side. Results: T2 3-2-1 and T5 4-1-1 against single-stage (never 5 of 6). Against the cut single-stage (the submitted TASK A), from `results/listening/t5_sections.csv`, `t6_select.csv`:

   | Session | Plan-fill side | vs cut single-stage |
   |---|---|---|
   | T5 | chord-relative fill, uncut, no penalty | 2-1 |
   | T6 | chord-relative fill, cut, penalty 2 (challenger, rule 8 of 10) | 6-4 (clear only 3-4) |
   | T5 + T6 | chord-relative fill | 8-5 |
   | none | current fill | never paired |

   Fill choice: chord-relative vs current 1-1 (T5) and 3-1 (T6). State plainly: close, a slight lean to plan-fill, below the rule, so TASK A stays single-stage; the 8-5 pools two set-ups; time ran out for more pairs. Open lead: chord-relative fill under a longer-trained plan model (`plan_rep_24k/loss_0.40`), generated and released for others to try, not listened to (`hw2_extra_plan_fill.zip`). Lessons: the fill trained far past its lowest held-out loss sounded better (T4 2-0).
9. **Harmony analysis.** Melody on chord tones and semitone clashes, real vs systems (`results/harmony.csv`, `clash_sides.csv`, `t6_*.csv`): current fill 0.646 chord-tone median vs real 0.712; chord-relative fixes it (0.707) but clashes double; the excess clashes are on the piano side (2.3-3.0x real vs melody 1.3-1.6x), which motivates a piano-only penalty; penalty 2 cuts piano-side clash time from 0.0056 to 0.0022. Listener comments that matched the MIDI (pieces 16, 18).
10. **BONUS TASK.** Bar-interleaved melody/piano tokens, decoder-only (option A), inference forces `Bar Mel <melody> Acc` per bar. CA and OA table vs baselines (`results/bonus.csv`, `bonus_baselines.csv`: ground truth CA 0.7625; melody an octave down 0.30). Ablation (`bonus_ablation.csv`): look-ahead and chords variants; the chords row with its transposition bug stated, or retrained (open). Melody vs piano chords: BONUS piano 0.46-0.51 vs real 0.72 (`bonus_harmony.csv`).
11. **How the listening tests were run.** Screenshots of the rating page (`results/figures/listening_page_desktop.png`, `listening_page_phone.png`: blind A/B with A, B and A+B playback, preference -2..+2 with reasons, a section duel per 8 bars, piano roll, comments; no model names on the page). Blind, sides balanced and alternated, rules written before rating (`docs/status.md` history), one listener (the author), 2-10 pairs per decision; order bias check (A 17, B 23, p 0.43); a feature-based grade predictor scored 59-61% out of sample, so decisions rest on listening only.
12. **Limitations and lessons.** Single listener, small samples; held-out loss vs listening; harmony still the main gap for plan-fill; what would be next (more listeners, chord-aware augmentation, a stronger BONUS harmony model). Citations: TA tokenizer and evaluation code, MusDr, PopMAG (CA/OA), POP909.
13. **Appendix.** Commands to reproduce (README), every listening session and its rule, the archived results not used.

## Open before writing slides

- **Slide tool:** decided: the claude.ai Slides type, same look as HW1, PDF downloaded by the user.
- **BONUS submission model:** decided: `bonus_ppb16/last`, `--top_p 0.95 --grammar`.
- **`abl_chords`:** state the bug (no reply; default).
- **T6:** done (slides 2, 6, 8). T7 final check only if time allows.
