# Course slides: notes for HW2

Read 2026-10-16 from `slides/` (not in git). Page numbers are slide numbers. "Img" lines describe figures whose content is not in the slide text. Each section ends with what it means for HW2. Tools and papers we use or dropped: `docs/references.md`.

## Lecture 05: MIDI Generation (94 slides)

### Representations (p2-24)

- Score vs performance: a MIDI score is on the metrical grid with no dynamics; a MIDI performance has expressive timing, tempo changes and velocity (p22-23). Musical time (1/4, 1/16) suits scores, physical time (ms) suits performances (p13, p21).
- Piano roll = image-like (time x pitch, fixed shape); MIDI events = text-like and can take extra tokens such as chord and key (p14-20). MIDI ticks per quarter note (PPQN), commonly 120 (p21).
- p24 libraries: miditoolkit (ticks), pretty_midi (seconds, piano rolls), pypianoroll, mido (low-level messages), music21 (analysis).

### Tokens (p26-54)

- Music vs text (p39): overlapping notes need a time representation, notes have several attributes, mid-level features (chords, groove), music is long (over 4k tokens), form (verse, chorus).
- MIDI-like (Performance RNN, p40): 128 note-on, 128 note-off, 100 time-shifts of 10 ms, 32 velocity bins. MuseNet (p41-45): `piano:v72:G1` and `wait` tokens; img note "No bar lines!".
- REMI (Pop Music Transformer, p46-49): `Bar` tokens, position within the bar (16 bins), tempo, chord (60 types), note duration instead of note-off, velocity (32 bins). Img p49: comparison table MIDI-like vs REMI. p50: REMI suits scores and steady-beat pop, and bar tokens keep multitrack parts in sync; MIDI-like suits expressive classical performance.
- p51-54: MidiTok library (REMI, CP, BPE, augmentation); BPE for music (Fradet et al., EMNLP 2023).

### Generation (p56-80)

- p58: RNNs for short sequences; "usually, a 12-layer Transformer decoder (with about 40M learnable parameters) works okay".
- p59-64: Music Transformer (relative attention, minute-long piano performances, accompaniment for a melody), MuseNet (72 layers), Pop Music Transformer. Img p63: user study, REMI wins 103:49 against Music Transformer (Wilcoxon p 5.6e-5); table of beat STD, downbeat STD, downbeat salience per token variant with real data as the last row.
- p65 DIY steps: task and data, tokens, model, train until loss is low enough, inference, listen, evaluate.
- **p73, train:** "Training loss around 0.30 gives better result"; the model "overfits and generates overly repeating content when loss is nearly 0"; "keep several checkpoints and listen to them to choose one"; "no need to consider validation data" unless the model is conditional (encoder/decoder).
- p74-77: greedy, beam, random, top-k, nucleus (top-p) sampling, temperature (above 1 for diversity). p75: teacher forcing in training vs sampled tokens at inference = exposure bias.
- **p78-79, choosing sampling settings:** they change results "a LOT" and papers rarely report how they were picked. Compose & Embellish searched temperature {1.2, 1.3, 1.4} x top-p {.95, .97, .98, .99} for the mean perplexity (by the model itself) closest to validation real data and chose 1.2 / .97. Img p79: entropy histograms of sampled tokens vs validation set: t0.8 p0.8 "too conservative" (JS 0.192), t1.5 p0.95 "too chaotic" (JS 0.536), t0.95 p0.95 "just right" (JS 0.057).
- p80: FluidSynth and midi2audio to listen.

### Evaluation (p83-94)

- p83: subjective MOS is more convincing; objective metrics "may not reflect human perception", good for early experiments. Img: C&E MOS table (coherence, correctness, structureness, richness, overall; real data 4.13-4.44).
- p84 aspects: correctness, coherence (over time and across tracks), structureness (recurring motifs, phrases, sections, development), richness, controllability.
- p85-90, MusDr (Jazz Transformer paper): pitch-class histogram entropy H (1 or 4 bars; clear tonality = low), grooving pattern similarity GS (binary onset vector per bar, 1 - mean XOR between bars), chord progression irregularity CPI (% unique chord trigrams), structureness indicator SI from the fitness scape plot (SI_3^8, SI_8^15, SI_15 for short, mid, long).
- **p91, table by loss level:** columns Model A at loss 0.80/0.25, Model B at 0.80/0.25/0.10, Real; rows H1 (real 1.94), H4 (2.87), GS (0.86), CPI (40.4), SI short/mid/long (real 0.36/0.36/0.35, models 0.10-0.27). Img: scape plot of the best model B piece is faint below 15 s; the real piece shows dark triangles up to about 40 s.
- p92-94, MusPy metrics: empty beat rate, empty measure rate, groove consistency, pitch-class entropy, pitch in scale rate, polyphony, scale consistency, and more.

### For HW2

- **Checkpoint choice (p73):** the course rule is train loss around 0.30, several checkpoints, listen, no validation needed for unconditional generation. This supports picking the TASK A checkpoint by training-loss level plus listening (our `loss_0.40`, `loss_0.30` checkpoints) and explains the spec's "Loss" column. It does not settle `loss_0.40` vs `step_10000`; listening does.
- **Sampling choice (p78-79):** pick temperature and top-p by matching the sampled-token NLL (or its entropy histogram) to the model's NLL on held-out real songs. `eval_generated.py` already writes the sampled-token NLL, so the TASK A sampling grid can be read this way.
- **Report table format (p91):** H1/H4/GS (plus CPI, SI if available) by loss level next to a Real column.
- **Structure metrics:** SI and CPI are in MusDr; we only use H1/H4/GS from the TA script. Whether the cloned MusDr computes SI from our files is unchecked.
- **Model size (p58):** our 12-layer, 39.2M-parameter GPT-2 is the size the lecture names.
- **Silence:** MusPy's empty beat and empty measure rates are standard names for what our silence stats measure.

## Lecture 06: Symbolic Music Generation 2 (110 slides)

### Token design (p10-31)

- **p11-13, long sequences:** REMI and MIDI-like use several tokens per note. Most pre-2020 models used a 512-token window; a pop piano cover can reach 5k tokens. Img p13: table of models and windows (Music Transformer 2048, MuseNet 4096, Pop Music Transformer and others 512 with Transformer-XL). Segmenting songs makes long-term patterns hard to learn.
- **p14-20, Compound Word (CP) Transformer:** groups a note's tokens (pitch, duration, velocity) into one step with several output heads. Img p14: Transformer-XL 4-17 GB, 3-7 days, MOS 3.0-3.3 vs CP 4 GB, 1 day, 8x real time, MOS 3.3. Img p19/20: rows named by loss level, `REMI + XL@0.44`, `REMI + XL@0.27`, `CP + linear@0.25`, with "Training data" and "Training data (randomized)" as reference rows.
- **p21, sampling per token type:** img table of CP's temperature and top-p per type: track 1.0/0.90, tempo 1.2/0.90, position/bar 1.2/0.90, chord 1.0/0.90, pitch 1.0/0.90, duration 2.0/0.90, velocity 5.0/1.00, family 1.0/0.99; REMI overall 1.2/0.90.
- **p22-24:** Multitrack Music Transformer (one head per field incl. instrument), Nested Music Transformer (sub-decoder per compound token; slower, better dependency), H-Net (dynamic chunking).
- **p25, token order for multitrack:** img: vertical parsing (all tracks at each onset), horizontal parsing (whole track, then next track), and "a natural way": bar by bar, track by track inside each bar.
- **p26-28:** time as position + delta tokens (MIDI-GPT); Moonbeam predicts delta time but takes absolute time as input; pitch as pitch class + octave.
- **p29, functional representation:** img: REMI vs functional token rows; functional adds a `Key` token and writes melody and chords as scale degrees (Roman numerals) relative to the key, C major and c minor mapped onto one Roman row.
- **p30, chord tokens:** degree, tonality root and mode, octave, extension (inversion, 7ths, sus), duration.
- **p31, structure tokens (phrase-length designated generation):** img: the user types a form like `i4 A8 ...`; the sequence gets `Phrase (i)` and `Bar Countdown (4..1)` tokens each bar, and the model writes the music.

### Conditioning and structure (p32-58)

- **p33/57, conditioning methods:** Theme Transformer = theme token sequence via cross attention; MuseMorphose = embedding sequence via in-attention; Compose & Embellish = lead sheet tokens via an interleaving prompt. p57: decoder-only models are easier to train than encoder/decoder; `Bar` segmentation is useful; "special care might be needed to avoid the conditions being ignored".
- **p34, why prompt conditioning is weak:** it cannot guarantee the prompt repeats or develops, and "the influence of the prompt would diminish over time and the generated music drifts away from its beginning".
- **p35-38, Theme Transformer:** themes found automatically (contrastive melody embedding, DBSCAN, largest cluster); `THEME-START`/`THEME-END` tokens; the model decides when to enter a theme region; cross attention is switched on only inside theme regions. Img p35: 24-bar piano roll, theme bars shaded pink at bars 1-2, 11-12, 16-17, 19-20.
- **p39, evaluation:** img plot of melody-embedding distance to the start vs bar (0-62): prompt-based rises to about 1.4 and stays; Theme Transformer and the original piece move between 0.2 and 1.2, returning low at theme bars. Img table (higher/lower better): pitch-class consistency (real .65), melody inconsistency (real .09, baseline .33, Theme T .13), grooving consistency (real .74), theme inconsistency, theme uncontrollability.
- **p40:** strength: repeats material; weakness: no development, structure not controllable. Question: generate a skeleton first, then fill details?
- **p41-44, MuseMorphose:** bar-wise encoder, per-bar attribute embeddings for rhythmic intensity and polyphony (bins per bar, img shows values like 3 4 ... 7), in-attention conditioning. p42 "The condition might be ignored!": happens when the decoder is strong or autoregressive context reveals much; remedies: weaker decoder, more relevant or understandable condition, stronger conditioning mechanism; a narrow bottleneck matters. Img p44: metrics fidelity (chroma and groove similarity), control (correlation of attributes), fluency (PPL), diversity.
- **p45-47, VQVAE Transformer (unpublished):** bar codes as a blueprint, then a decoder. Img p46, structure measured by note n-gram stats: unique % (short 3-6 notes, mid 7-15, long 16-40) real 42.7/60.9/75.5, VQ 45.3/64.1/77.0, unconditional 9.3/15.2/27.1; reappearance interval in bars real 16.6/23.7/29.2, unconditional 2.7/3.6/4.7. Plagiarism check: longest common note string, generated vs its reference song 6.24 +/- 1.98 notes, random real pairs 3.79 +/- 1.49.
- **p48-54, Compose & Embellish:** modular beats end-to-end: generate melody first, then piano. p49: p(C) structure -> p(M|C) melody -> p(X|M) full performance; lead sheet = melody + chords. p50: lead sheet model writes phrase form (e.g. A B A A), chords, melody onset/pitch/duration; performance model writes accompaniment, dynamics, beat-level tempo. Img p50: 32 bars labelled A A B C / B C A A. p51: performance model input interleaves M(1) X(1) M(2) X(2), "keeps melody context nearby"; lead sheet model pretrained on general MIDI. Img p52: fitness scape plots (triangles) for C&E, CP Transformer and real data; real and C&E show darker blocks (more structure). p53: structureness SI short/mid/long (real 43.8/43.1/34.8, C&E 36.8/35.1/25.8, CP 32.5/29.9/17.9) and melody-line diversity DN; naive 1-bar repeats score SI 83.6/88.3/75.7 with DN 1.2, so high SI alone is not good. p54: listening test, 5-point Likert on coherence, correctness, structureness, richness, overall (real 4.42/4.13/4.44/4.24/4.40, C&E 3.53/3.11/3.36/3.29/3.18, CP 2.38/2.49/2.33/2.64/2.33).
- **p58, recommended:** piano: Pop Music Transformer, CP Transformer, Compose & Embellish; multitrack: Multitrack Music Transformer, Anticipatory Music Transformer.

### Image-based approach (p59-77)

- **p61-70:** piano roll (88 pitches x 96 steps per bar); non-autoregressive infilling (Coconet, Bach Doodle); Polyffusion (diffusion on piano rolls: unconditional, melody given accompaniment and back, inpainting, chord and texture control); MuseGAN (5 tracks, 4 bars, shared and private z). Img p68: MuseGAN samples at steps 0/700/2500/6000/7900 going from noise to sparse notes.
- **p71-75, whole-song hierarchical generation (Wang et al., ICLR 2024):** four levels: form (key changes, phrases) per measure, reduced lead sheet per beat, lead sheet per step, accompaniment per step; cascaded diffusion, each level conditioned on the one above. Img p72: form strings like `(i8)(A8B16A8B16)(b6)(B14)(o2)` and `(i4)(A4A4B4X5)(b4)(A4b5B4X4X5)(o2)` (i intro, o outro, b bridge, A/B/X phrases with lengths), plus tonic and scale rows per measure.
- **p76-77, onsets:** a piano roll cannot tell one long note from two repeated notes; fix with an extra onset channel.

### Beyond MIDI (p78-110)

- MusicXML (enharmonics, voices; Score Transformer), ABC (folk-rnn, LLMs, NotaGen), PDMX dataset. Img p89: dataset table lists POP909 as MIDI, 60 hours, 909 songs. MIDI-LLM tokens: onset time, duration, instrument-pitch (27.5K tokens). p109: `html-midi-player` web component with a piano-roll visualizer.

### For HW2

- **Plan-fill matches the lecture's direction:** "skeleton first, then details" (p40, p47), melody first then piano (p48-49), bar-interleaved melody and piano (p51), and a typed form string (p31, p72). Our `--form I4A8B8A8E4` is the same idea as p31/p72; the report can cite Compose & Embellish and Wang et al.
- **"The condition might be ignored" (p42, p57) is our repeat-label finding:** the fill model copies under teacher forcing but writes the asked repeat 4.3% of the time when sampling. The slide's remedies map to options: a stronger mechanism (our `RepeatGuide` logit bias is one), a more understandable condition (point at the source bar's notes, not just a label).
- **TASK B prompt drift (p34, p39):** the lecture expects prompt influence to fade; our `prompt_fade.py` measures that, and p39's plot (distance to the start per bar, with the original piece as reference) is a good figure format.
- **Structure measures to borrow (p46, p53):** unique % of note n-grams at short/mid/long lengths, reappearance interval in bars, SI with a naive-repeat baseline, longest common note string for copying. These could fill the gap where H1/H4/GS do not separate anything (finding 1).
- **Listening rubric (p54):** our rubric's five aspects (coherence, correctness, structureness, richness, overall) are this slide's; it reports real data at about 4.4 as the anchor.
- **Per-type sampling (p21):** CP samples duration at temperature 2.0 and velocity at 5.0. Our sampler uses one temperature for all types; per-type temperature is a cheap experiment.
- **Checkpoint naming by loss level (p19-20):** `REMI + XL@0.44` vs `@0.27` matches our `loss_0.40` style and the spec's "Loss" column.
- **Repeated same-pitch notes (p76):** the long-note vs repeated-note ambiguity is the same family as our one-key rendering fix.

## Lecture 06b: Transformer (19 slides)

- RNN keeps one latent vector; a Transformer keeps one per past token, y_t = f(y_t-1, [h_t-1 ... h_t-L]), and needs positional encoding (p3-4).
- Decoder-only is used for unconditional or prompt-based generation; encoder/decoder for sequence-to-sequence (p5, p16).
- p7-14 (from Jia-Bin Huang's video): token embedding, Q/K/V attention, softmax(K^T Q / sqrt(d_k)), multi-head, FFN = W2 ReLU(W1 x + b1) + b2, sinusoidal positional encoding (relative shift is a linear map), pre-LayerNorm residual blocks. p17: family tree of encoder-only, encoder-decoder, decoder-only models. p18-19: links on RoPE and FlashAttention.
- **For HW2:** background for the report's model slide: GPT-2 is a decoder-only Transformer with learned positions, used for unconditional (TASK A) and prompt-based (TASK B) generation.
