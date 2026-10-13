# TA scripts (HW2)

These files come from `HW2_scripts.zip`, provided by the HW2 TA (Bo-Rui Chen) for NTU Deep Learning for Music Analysis and Generation, 2026. They are kept here so the pipeline runs as submitted.

Changes from the original:
- `tokenization.py` lines 139 and 146: pass `POS_PER_BEAT` to `time_to_grid`. The original always used 16 positions per beat on the grid, so `--pos_per_beat` other than 16 produced `Pos_` tokens outside the vocabulary.

`eval_basic.py` imports MusDr from `./MusDr` (not committed). Clone it next to the scripts:

```bash
git clone https://github.com/slSeanWU/MusDr.git
```

The TA's original README follows.

---

# Usage For Each Scripts
### `tokenization.py`
This script is for processing the POP909 data and converting the MIDI messages into self-defined tokens
```bash
python tokenization.py --dataset_path ${POP909 training set path} --tok2idx_output ${token dictionary output path} --sequences_output ${tokenized song sequences output path} --pos_per_beat ${quantization setting}

# e.g. python tokenization.py --dataset_path ./POP909/training --tok2idx_output ./tok2idx.json --sequences_output ./all_song_sequences.json --pos_per_beat 16
```

### `eval_basic.py`
This script is for running objective metrics for `Basic Task a.`
```bash
python eval_basic.py --tok2idx_path ${token dictionary path} --sequences_path ${tokenized song sequences path} --output_csv ${results output path}

# e.g. for calculating the real data scores: python eval_basic.py --tok2idx_path ./tok2idx.json --sequences_path ./all_song_sequences.json --output_csv ./unconditional_metrics.csv
```

### `eval_bonus.py`
This script is for running objective metrics for `Bonus Task`
```bash
python eval_bonus.py --prompt_dir ${melody prompts folder path} --gen_midi_dir ${generated MIDIs folder path} --output_csv ${results output path}

# e.g. python eval_bonus.py --prompt_dir ./POP909/prompts/bonus --gen_midi_dir ./bonus_task_gen_midis --output_csv ./conditional_metrics.csv
```