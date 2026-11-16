# HW3 report narrative (11-16)

Intervals are 95% paired bootstraps over prompts, no multiplicity correction. No test caption text is quoted; test appears only as per-group counts. Spec (`Homework_3.pdf`, Scoring): 100% report, performance does not affect the score, insight is valued, extra credit for MQAScore, one more method, ablation, clarity. Data files live under `results/analysis/` unless a full path is given.

## 1. Story in five sentences

Stable Audio Open follows some caption words (instrument, voice, acoustic versus electric, jazz) and not others (recording quality, country, tempo at best weakly), and it renders mood and loudness words through arrangement bundles such as drums and groove (`edit_effects.csv`, `edit_spillover.csv`). The test captions lean on words the encoders hear poorly, so the size of any gain depends on which words a caption uses (`caption_vocab_groups.csv`, `attribute_probes_cv.csv`). Each metric reads a different part of the caption: CLAP sees a lost instrument or voice, MQAScore dilutes it in a caption mean, KL never reads the caption, and the three judges pick the same seed at chance rate (`ablate_dev_scores.json`, `ranker_compare.csv`). Seen through the study, CFG rescale strengthened what the model already follows (dev +0.028, test +0.010), best-of-3 is a judge-defined, partly circular gain, and the word-level methods (followed-word guidance, probe-in-the-loop) gave nothing beyond extra guidance weight or an extra seed. The report claims one honest gain, one qualified gain, several explained negatives, and a rule for the next round: split captions by the word classes the model follows before choosing a method.

## 2. Outline (id, claim, points, figure, sources)

Existing figures (`A` in `report/deck/build.py`): `hubness`, `cfg_sweep`, `trajectory`, `forest`, `mqa_attr`, `listening`, `fd_floor`, `roadmap`. Anything else is a new figure.

### Part A: question and baseline

1. `cover` (kept). "Does the model follow the caption words, and what does that mean for how we improve and judge it?" Name and ID, repo and release links. No figure.
2. `summary` (rewritten). "The model follows some caption words and not others, and that sets what any method can gain."
   - Study first, then methods: rescale dev dCLAP +0.028 [+0.020, +0.035], dKL -0.119 [-0.170, -0.068]; test +0.010 [+0.005, +0.016]; best-of-3 partly circular; word-level methods add nothing beyond weight.
   - Submission: rescale 12, best-of-3 ranked by CLAP checkpoint 89.98, test dCLAP +0.040 [+0.027, +0.051].
   - Sources: `docs/analysis_review.md`, `findings_table.md`.
3. `setup` (merged with `evalsetup`). "Tuning used 100 dev captions that exclude every retrieved test clip, and every comparison is paired."
   - Dev vs test table kept (mean words 48.1 vs 28.4; quality words 41% vs 1%; vocal tags 50 vs 1).
   - Shared noise makes intervals 0.73x as wide (CLAP SD 0.065 vs 0.090) (`docs/analysis_review.md`).
   - Test rule box: test ranking uses captions and our own audio only, scored by a second checkpoint (89.98, not the scorer 90.14); predictions written before results (`docs/predictions.toml`).
   - Sources: `retrieval/caption_style.csv`.
4. `retrieval` (kept). "Plain top-1 retrieval returns only 59 distinct clips for 100 captions." Figure `hubness`. Sources: `results/retrieval.json`, `retrieval/hubness_reduction.csv`.
5. `cfg` (merged with `baseline`). "Our CFG matches the TA reference step for step, and the test baseline scores CLAP 0.31."
   - TA check max diff 0; test CFG 7: CLAP 0.314, KL 0.950, FD-VGG 8.16; FD floor for 100 real clips: FD-VGG 1.9, FD-MERT 21.
   - Figure `fd_floor`. Sources: `text_to_music/cfg.py`, `findings_table.md`, `docs/findings.md`.

### Part B: caption-word study (the lead)

6. `capask` (new; merges "what captions ask for" and "what encoders hear"). "Test captions lean on words the encoders hear poorly."
   - Dev names voices (male 30, female 16, choir 8) and low quality (38); test names no voices and 3 low-quality; test leans on shimmering/warm/lush 46 vs 12 and dark/tense 24 vs 3; median words 21 vs 48.
   - Probe AUC (MERT/CLAP): voices, guitar, drums 0.90-0.93; mood, tempo, dynamics about 0.71-0.77; shimmering/warm/lush 0.65-0.67; high quality 0.59-0.60; context 0.60-0.63; ambient 0.58-0.62.
   - Detector check before claims: beat BPM fails, onset-based reader AUC 0.84 on a fresh sample (`dsp_detectors.csv`); readers count only at AUC >= 0.65.
   - Vocals: dev vocal attribute 0.22 generated vs 0.49 real; test has 1 vocal caption, so vocals hardly touch test.
   - New figure: probe AUC bars by group next to dev/test counts (`attribute_probes_cv.csv`, `caption_vocab_groups.csv`); backup `figures/genre_confusion.png`. Also `retrieval/caption_style.csv`, `probe_validity.csv`, `vocal_split_dev.csv`.
7. `follow` (new). "Instrument, voice and jazz words are followed; recording, country and tempo words are not; a word moves a clip about as much as a seed."
   - Pooled moved/n (MERT / CLAP probe): guitar to piano 25/28, 25/28; voice 21/26, 20/26; acoustic to electric 14/17, 12/17; insert jazz 39/50, 36/50. Insert country 27/50, 27/50; low quality to studio 21/35, 17/35 (chance).
   - Tempo: probe 10/14 (p 0.09), onset reader 9/14 (p 0.21), beat BPM 5/14; only an all-feature reader moves (12/14): "weakly at most".
   - Gate: unchanged captions through the edit path reproduce `dev__rescale12` bit-identically (`null_edit.json`).
   - Controls: meaningless edits change the clip 0.68-0.77 of a seed change, real edits 0.77-1.27; groups clear of zero 2-3 vs up to 8-9 (`edit_controls.csv`).
   - Caveat: about 200 intervals, some clear by chance; name only large consistent cells.
   - New figure: moved/n per axis (`edit_effects.csv`, `edit_controls.csv`); optional mel pairs for two axes.
8. `bundles` (new; spillover plus bundle cut). "Mood and loudness words act through bundles; cutting the loud bundle works, cutting the sad bundle changes the meaning."
   - Spillover in seed SD, clear of 0: "loud" brings groove +0.78, drums +0.64, male voice +0.53, happy +0.41; "sad" strips drums -0.61, groove -0.51; acoustic to electric drops sad -0.93 and adds synth +0.65 (`edit_spillover.csv`).
   - Spillover tracks co-occurrence in captions and real audio (Spearman 0.32 each; the two correlate 0.89; 36 clear pairs, sign agreement 0.81) (`bundle_origin.json`); the audio side uses the same probes.
   - Bundle cut (caption weight kept at 12): loud bundle arm lowers drums -1.11 SD [-1.61, -0.63] and groove -1.11 [-1.60, -0.61] while the loud probe stays (+0.02 [-0.24, +0.28]); neutral arm does nothing. Sad: drums restored +1.71 [+1.05, +2.42] and CLAP sad score falls -0.66 SD [-0.95, -0.39] (neutral +0.01 [-0.27, +0.27]) (`bundle_guidance.csv`).
   - Caveat: drums probe partly reads level; level alone predicts 0.26 of the 0.64 SD change for loud, not the sad change (`probe_leakage.json`).
   - New figure: heat map (`edit_spillover.csv`) plus bundle-vs-neutral bars (`bundle_guidance.csv`).
9. `recording` (new). "Dropping or swapping recording words changes nothing but the roll; a followed word shows at once."
   - Swap low quality to studio: not followed by any detector (quality reader 8/12 and 18/35, MERT 4/12 and 21/35, CLAP 9/12 and 17/35).
   - Removing recording and setting sentences (29 dev captions): CLAP -0.012 [-0.052, +0.027], MQAScore +0.015 [-0.024, +0.051]; re-rolls like a seed (mean abs dCLAP 0.083 vs 0.075). Say "nothing but the roll"; power limit about +-0.04 CLAP.
   - Positive control, dropping a followed-word sentence: removed-group probe -1.09 SD [-1.49, -0.72] (81% of 43 groups down), CLAP -0.086 [-0.132, -0.041]. CLAP is not blind at n = 29 (this does not show it hears quality).
   - Cleaning run `rescale12_clean`: CLAP -0.003 [-0.016, +0.008], KL +0.013 [-0.057, +0.089].
   - New figure/table: `edit_effects.csv`, `ablate_dev_scores.json`, `clean_dev.json`.
10. `metrics` (new; hard slide 1). "A lost instrument or voice moves CLAP but not caption-mean MQAScore or KL: the metrics read different parts of the caption."
    - Removal of a followed word (29 captions): CLAP -0.086 [-0.132, -0.041]; MQAScore +0.004 [-0.042, +0.049]; KL +0.32 [-0.08, +0.68].
    - Per tag MQAScore does see it: removed voice -0.30 [-0.45, -0.16], removed instrument -0.15 [-0.35, +0.04]; kept instruments rise +0.08 [+0.02, +0.16] (77 tags) (`ablate_mqa_tags.csv`).
    - MQAScore asks only genre, instrument, mood, vocal, context; instrument tags are 44% of tags, 47% of variance across clips, 58% across seeds (`mqa_drivers.csv`).
    - Degraded clip: KL +0.075 [+0.032, +0.118] (predicted down, missed), MQAScore -0.021 [-0.035, -0.009] (`knowledge.md`).
    - Real vs generated: mean CLAP 0.302 vs 0.299, MQAScore 0.646 vs 0.546 (paired +0.101 [+0.064, +0.136]).
    - New figure: metric x intervention grid (`ablate_dev_scores.json`, `ablate_mqa_tags.csv`, `knowledge.md`).
11. `bridge` (new; hard slide 2). "Each method acts on a different part of the caption, so the study says what each can gain."
    - Table, one row per method: what it changes / what the study says / what it gave. Scale, rescale, APG strengthen followed words (CFG 12 +0.015, rescale 12 +0.028, test +0.010). Interval: late steps matter (-0.027, -0.075), first 21 steps free. Tags lose the followed sentences (-0.051). Best-of-3: judge-defined. Followed-word and probe loop: nothing beyond weight or a seed.
    - One reading, untested: test captions lean on words guidance cannot strengthen, so dev +0.028 becomes test +0.010 (caption margin +0.015 to +0.006, `clap_margin.csv`).
    - New figure: the table; `forest` as backup.

### Part C: methods through the study

12. `cfgsweep` (kept). "Higher scale raises CLAP and loudness; scale 3 is clearly worse." cfg3 -0.029, cfg12 +0.015; FD-MERT 41.0 / 43.5 / 44.6; loudness -18.8 / -17.4 / -16.7 dBFS; clip fraction is not a clean signal. Table. Sources: `findings_table.csv`, `extra_metrics.csv`.
13. `guidance` (kept). "Guidance is large early and small late, yet only the late steps matter." Off for sigma > 110 +0.000 [-0.003, +0.005]; sigma <= 2.6 -0.027 [-0.034, -0.019]; sigma <= 12 -0.075 [-0.094, -0.056]. Figure `trajectory`.
14. `prompts` (kept, plus one study line). "Tag prompts are worse on metrics that never read the caption." dCLAP -0.051 [-0.074, -0.027], dKL +0.43 [+0.28, +0.63], dFD-VGG +0.90 [+0.24, +1.59]; human tags -0.041; tags plus caption -0.001. Study line: tags drop the sentences that carry followed words.
15. `rescale` (merged with `apg`). "CFG rescale raises CLAP by 0.028 and lowers KL over three seed pairs; APG does not separate from CFG 12."
    - Pooled dev: dCLAP +0.028 [+0.020, +0.035], dKL -0.119 [-0.170, -0.068]; FD-MERT 41.0 / 41.2 / 41.5 below all cfg7 seeds.
    - Ablations: rescale 7 -0.003 [-0.012, +0.004] (needs high scale); scale 18 vs 12 +0.002 [-0.010, +0.015]; phi 1 vs 0.7 +0.006 [-0.004, +0.015], FD-VGG -0.305 [-0.529, -0.013] (prediction missed); valid-frame std -0.003 [-0.013, +0.008], dKL -0.055 [-0.115, +0.007].
    - APG 12 vs CFG 12 -0.004 [-0.018, +0.011]; APG fails FD-VGG (+0.36 [+0.11, +0.60]).
    - Figure `cfg_sweep`. Sources: `findings_table.csv`, `docs/analysis_review.md`, `knowledge.md`.
16. `bestof3` (merged with `clapmargin`). "Best-of-3 inflates its ranker's own CLAP, not the KL gain."
    - Vs seed mean: same-checkpoint dCLAP(90.14) +0.047, cross-checkpoint +0.020; dKL -0.104 [-0.156, -0.050] and -0.088 [-0.149, -0.027].
    - Margin: rescale vs cfg7 +0.015 [+0.009, +0.021]; best3 (89.98) vs rescale12 margin dev +0.003 [-0.009, +0.015], test +0.0132 [+0.0025, +0.0234]; same-checkpoint +0.031 [+0.020, +0.041]. Best-of-3 CFG 7 MQAScore missed (-0.007 [-0.036, +0.019]).
    - Sources: `best_of_n_vs_seed_mean.csv`, `clap_margin.csv`, `knowledge.md`.
17. `rankers` (new). "The judges disagree on which seed is best, so best-of-N is defined by the judge."
    - Picks agree 31-43% vs 33% by chance; each ranker lifts mainly its own metric (dev dCLAP 89.98 +0.015, probe -0.002, MQAScore +0.003); only the 89.98 ranker also helps dev KL (-0.124 [-0.218, -0.038] vs -0.040, -0.041) (`ranker_compare.csv`).
    - Within-caption rank correlation: checkpoints 0.47 dev / 0.50 test; MQAScore-CLAP 0.19 / 0.24; probe agrees with neither on dev (-0.05, -0.03). No judge is "louder": every feature link |rho| <= 0.23 (`ranker_explain.csv`).
    - New figure: ranker table plus correlation panel.
18. `wordmethods` (new, overnight results). "Methods built from the study add nothing beyond guidance weight or an extra seed."
    - Followed-word guidance on CFG 12 (95 of 100 dev captions have a followed phrase, median 5): vs rescale 12 dCLAP +0.002 [-0.010, +0.014], dKL -0.093 [-0.164, -0.025], MQAScore +0.001 [-0.016, +0.017]; 23.6 vs 16.4 s/clip. Scale 16: vs rescale 12 +0.005 [-0.005, +0.015]; KL vs cfg7 followed -0.177 [-0.285, -0.066] vs scale 16 -0.174 [-0.286, -0.061]; followed vs scale 16 -0.003 [-0.017, +0.010]. The gain is weight, not words.
    - Same total weight 12: dCLAP -0.006 [-0.018, +0.007], MQAScore +0.013 [-0.005, +0.029] (a small trade, both include 0).
    - Probe-in-the-loop: round 1 dev dCLAP -0.002 [-0.015, +0.011], test +0.010 [-0.001, +0.022]; round 2 -0.001 [-0.009, +0.006], dKL -0.024 [-0.071, +0.018]; kept fraction 0.303 (predicted 0.35-0.65, missed).
    - New figure: forest-style rows vs rescale 12 (`findings_table.md`, `knowledge.md`, `loop_dev.json`, `followed_dev.json`).

### Part D: MQAScore

19. `mqaimpl` (kept). "MQAScore runs on one 16 GB GPU and passes its sanity checks." bf16 plus offload; matched 0.700 vs mismatched 0.359 (11 of 12); reruns bit-identical; not reference-checked. Sources: `scripts/evaluation/mqa.py`, `docs/findings.md`, `docs/mqa_notes.md`.
20. `mqaresults` (merged with `vocals`). "MQAScore separates real from generated, and vocals are the main gap."
    - Real 0.646 vs 0.546; gaps genre +0.244, mood +0.172, vocal +0.271, context +0.055, instrument +0.025 (not separable); length bias (Spearman -0.24 to -0.41), no loudness bias; real vocal captions lose -0.332 on instrument.
    - PaSST vocal presence 0.063 vs 0.173; rescale gain equal in vocal and other captions (+0.028, +0.027); vocal split misassigns about 3-5 of 100. Not checked: judge training overlap with MusicCaps.
    - Figure `mqa_attr`. Sources: `findings_table.csv`, `vocal_split_dev.csv`, `extra_metrics.csv`.

### Part E: choice, test, wrap-up

21. `devchoice` (kept, add test-rule box). "The pick followed a rule fixed before dev, amended once after." Two rule definitions with times (11-15 11:50, 16:15), both after dev was seen; rescale 12 kept because only it had 3 seeds and a best-of-3 pool; scale 18 and phi 1 look at least as good. Figure `forest`. Sources: `knowledge.md`, `docs/analysis_review.md` item 10.
22. `testresults` (merged with `devtest`). "On test, rescale 12 gains +0.010 CLAP alone and +0.040 with best-of-3; all test predictions met; the gain is a third of dev."
    - Seed 0 dCLAP +0.010 [-0.000, +0.021], dKL -0.021 [-0.101, +0.053]; best-of-3 +0.040 [+0.027, +0.051], dKL -0.067 [-0.170, +0.031]; MQAScore 0.651 (cfg7), 0.658 (rescale 12), 0.672 (submission).
    - Show seed 0 next to best-of-3; call the best-of-3 CLAP gain selection-inflated. Smaller on test: shorter captions (28.4 vs 48.1 words), retrieved references, hub references (KL 0.82 / 0.65 / 0.75 vs 1.04 / 1.07 / 1.05); wrong reason for flat KL (pooled non-vocal dev KL -0.086 [-0.161, -0.015]).
    - Sources: `findings_table.md`, `docs/predictions.toml`, `docs/analysis_review.md`.
23. `listening` (kept). "Blind listening leans toward the picks, but 16 pairs cannot show it." Session 1: 9 / 5 / same 2 (p 0.42); session 2: 9 vs 6, same 1 (p 0.61); 13 of 16 needed; side B 20 of 29 (p 0.06); one listener. Reasons from 31 notes: instrument 10, vocal 8, mood 6 (the same word classes as slides 6-7). Figure `listening`.
24. `missed` (merged with `pilots`). "Twelve predictions missed, and each miss changed a claim."
    - Resolved: 35 met, 12 missed (`knowledge.md`); all test predictions met.
    - Misses that changed the story: rescale 7, phi 1, degraded-clip KL, probe ranker CLAP, ablation overshoot (CLAP -0.025 [-0.041, -0.010] over 100 captions), followed vs scale 16, round 2 kept fraction.
    - Left out (torch.compile, CFG++, negative prompts, SDEdit/fine-tuning, Qwen3-Omni MQAScore) and what we would do differently (split captions by followed class before tuning; choose methods by the class they can move; more than one judge; GPU for controls first).
    - Pending predictions (interval early12, APG projection/eta1, compose7, tags and interval_late MQAScore, best-of-5, followed_sentence): say "not run" in one line, no placeholders.
25. `reproduce` (merged with `references`). "Reproduce: three seeds, rescore with 89.98, select, package; every code base, model and paper is cited." Seeds `id + 1000 x index`, `scripts/evaluation/clap_rescore.py`, `scripts/selection/select_best_of_n.py`, `scripts/submission/build_and_check_submission_zip.py`; picks in `results/test__rescale12_best3_c8998/selection.csv` (26 / 41 / 33); 41 of 100 test picks have a top-2 gap under 0.022 and can flip on other hardware; release `hw3` (publish waits for the user). References from `docs/methods_research.txt`. If the list is long keep `references` as slide 26.

## 3. Drop or merge

- Drop `roadmap` from the main deck (keep as backup); its message moves to `bridge`.
- Merge `evalsetup` into `setup`, `baseline` into `cfg`, `apg` into `rescale`, `clapmargin` into `bestof3`, `vocals` into `mqaresults` (remove the pending `caption_classes` box), `devtest` into `testresults`, `pilots` into `missed`, `references` into `reproduce`.
- Reason: free room for 7 new study slides and 3 new method-reading slides (`bridge`, `rankers`, `wordmethods`) at the same total length.
- Fix stale text: the `missed` caption (12 met, 3 missed; now 35 met, 12 missed, 12 pending) and the `summary` "four findings" cards.

## 4. Speaker notes for the three hardest slides

**Slide 10, `metrics` (why the metrics disagree).** We removed one sentence with a followed word from 29 dev captions and scored against the original caption. The removed-group probe fell 1.09 seed SD, so the model dropped the attribute. CLAP fell 0.086 because it compares one audio embedding with one embedding of the whole caption, and a missing instrument noun is not offset by anything else. MQAScore did not move (+0.004): it averages one yes/no answer per tag, the removed tag is about one of eight, and the kept instruments are judged more present once one part is gone (+0.08), so the loss hides in the mean. Per tag it does see it (voice -0.30, instrument -0.15). KL compares class distributions with the reference clip, never the caption, so it moves only when the class mix shifts; its spread at n = 29 is large. The reverse also holds: a degraded clip moves KL and MQAScore in opposite directions, and MQAScore never asks about recording words. So "which metric improved" says which part of the caption a method touched, and no single metric stands for "better". Hence every gain is paired with a second check, and mean CLAP (0.302 real vs 0.299 generated) is read with its margin and MQAScore. Sources: `ablate_dev_scores.json`, `ablate_mqa_tags.csv`, `mqa_drivers.csv`, `knowledge.md`.

**Slide 11, `bridge` (what the study says about our methods).** Scale, rescale, APG and the interval only change how hard the model pushes toward the caption. The swap test shows it follows instrument, voice, acoustic/electric and jazz words and renders mood and loudness through drum/groove bundles; recording words and "country" are not followed. So more guidance strengthens the followed part and its bundle; it cannot add a word the model ignores. This is one reading of why gains are small on dev (+0.028) and smaller on test (+0.010): test captions lean on texture and mood words the encoders hear poorly (shimmering/warm/lush 46 vs 12). That link is not tested directly. Tags failed because a tag list drops the sentences that carry followed words. Best-of-3 is where control sits, since a word moves a clip about as much as a seed does, but the three judges pick the same seed at chance rate and our ranker partly wins because it is a cousin of the scorer. We built two methods from the study, followed-word guidance and probe-in-the-loop; both gave nothing beyond extra weight or an extra seed (dCLAP +0.002 and -0.001; KL gain equals plain scale 16). This fits the study: followed words need no more weight, and ignored words cannot be pushed by the same knob. The one positive hint is the bundle cut: guiding away from the loud bundle removes drums and groove (-1.1 SD) while the loud probe stays, so bundle-level methods are the next direction; but for "sad" restoring the rhythm weakens the sadness (CLAP -0.66 SD), so some bundles are part of the meaning.

**Slide 24, `missed` (limitations).** One listener (the author), 16 pairs per session, side B chosen 20 of 29 decided pairs: listening is exploratory. Dev is 100 captions tuned on once; test is 100 captions with retrieved references picked by the scoring checkpoint, so test KL is not independent of CLAP and hub references make it easier. "Followed" means "moved the target probe or detector" with MERT and CLAP readers, so a failed detector and an ignored word look alike; the paired design and the AUC >= 0.65 rule narrow this but do not remove it. The swap test has about 200 intervals, so some clear by chance. The recording-word null has a power limit of about +-0.04 CLAP at n = 29. The drums probe partly reads level. The selection rule was amended once after dev was seen. Followed-word guidance and probe-in-the-loop ran on dev with one seed; best-of-5 and several held predictions did not run. We did not check whether the MQAScore judge saw MusicCaps in training. No multiplicity correction: cfg12 and APG 12 dCLAP do not survive Holm over 48 tests. We claim a gain only where the paired interval excludes 0, and we state each limit on the slide where the number appears.
