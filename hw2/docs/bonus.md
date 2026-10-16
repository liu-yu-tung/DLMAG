# BONUS TASK tokens

`src/hw2/bonus.py`: a decoder-only model writes PIANO for a given MELODY on bar-interleaved tokens.

- **Sequence:** `<BOS>`, then per bar `Bar Mel <melody notes> Acc <piano notes>` (`Mel` = melody block, `Acc` = piano block), each note `Pos Pitch Vel Dur` on the TA grid and bins. No tempo tokens: timing comes from `beat_midi.txt`.
- **Inference:** the `Bar Mel <melody> Acc` part of each bar is forced; piano notes are sampled until the model writes `Bar`. Generated notes are timed by the prompt's beat file.
- **Ablation variants** (stored in `variant.json` next to the vocabulary): `lookahead` puts bar b+1's melody in bar b's melody block, so the piano sees the melody one bar ahead; `chords` starts each bar with `Chd` and (`Pos`, `Chord_<root>_<quality>`) pairs from `chord_midi.txt`. `chords` reads the labels that the TA chord accuracy is scored against, so it is reported as an ablation, not the headline.
- **Models:** `abl_melody` (the headline run, tokens `data/tokens/bonus_ppb16`), `abl_lookahead` (`bonus_la`), `abl_chords` (`bonus_chd`); `scripts/bonus_ablation.py` trains and scores all three.
