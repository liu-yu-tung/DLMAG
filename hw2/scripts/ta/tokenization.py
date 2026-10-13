"""
Install the following dependencies before running this script:
- pathlib
- pretty_midi
"""

import json
import bisect
import pathlib
import argparse
import pretty_midi

def parse_args():
    parser = argparse.ArgumentParser(description="Process the POP909 dataset and convert it into a tokenized representation.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the POP909 dataset directory.")
    parser.add_argument("--tok2idx_output", type=str, default="tok2idx.json", help="Output path for the token-to-index mapping JSON file.")
    parser.add_argument("--sequences_output", type=str, default="all_song_sequences.json", help="Output path for the tokenized song sequences JSON file.")
    parser.add_argument("--pos_per_beat", type=int, default=16, help="Number of positions per beat (default: 16).")

    return parser.parse_args()

def get_beat_info(beat_orders):
    """
    Given a list of beat orders, returns a list of tuples (bar_index, beat_in_bar) for each beat.
    A new bar (a downbeat) starts when the beat order is 1.0.
    """
    beat_info = []
    bar_idx = -1

    # Find the indices of all downbeats
    downbeat_indices = [i for i, order in enumerate(beat_orders) if order == 1.0]

    # Determine beats_per_bar for the opening of the song
    if len(downbeat_indices) >= 2:
        beats_per_bar = downbeat_indices[1] - downbeat_indices[0]
    else:
        beats_per_bar = 4

    first_downbeat_idx = downbeat_indices[0] if downbeat_indices else -1

    if first_downbeat_idx > 0:
        bar_idx = 0
        for i in range(first_downbeat_idx):
            beat_in_bar = (beats_per_bar - (first_downbeat_idx - i)) % beats_per_bar
            beat_info.append((bar_idx, beat_in_bar))

    elif first_downbeat_idx == -1:
        first_downbeat_idx = 0

    beat_in_bar = 0
    for i in range(max(0, first_downbeat_idx), len(beat_orders)):
        if beat_orders[i] == 1.0 or bar_idx == -1:
            bar_idx += 1
            beat_in_bar = 0

        beat_info.append((bar_idx, beat_in_bar))
        beat_in_bar += 1

    return beat_info

def time_to_grid(time_sec, beat_times, beat_info, POS_PER_BEAT=16):
    """
    Maps exact seconds to a quantized measure and position grid.
    """
    b_idx = max(0, bisect.bisect_right(beat_times, time_sec) - 1)
    b_idx = min(b_idx, len(beat_times) - 1)
    bar_idx, beat_in_bar = beat_info[b_idx]

    b_start = beat_times[b_idx]
    b_end = beat_times[b_idx+1] if b_idx + 1 < len(beat_times) else b_start + (beat_times[-1] - beat_times[-2] if len(beat_times)>1 else 0.5)

    progress = max(0, min(0.999, (time_sec - b_start) / max(0.001, b_end - b_start)))
    pos = beat_in_bar * POS_PER_BEAT + int(progress * POS_PER_BEAT)
    return bar_idx, pos, max(0.001, b_end - b_start)

def create_vocabulary(dataset_path, MAX_POSITIONS):
    data_dir = pathlib.Path(dataset_path)

    # Extract chord information from chord_midi.txt
    for chord_file in data_dir.rglob("chord_midi.txt"):
        with open(chord_file, "r") as f:
            for line in f:
                parts = line.strip().split()  # [start_time(sec), end_time(sec), chord_label]

    # Define the token vocabulary here
    # You can modify here to create your own vocabulary (e.g. adding chords, changing ranges, etc.)
    VOCAB = {
        "Bar": ["Bar"],
        "Position": [f"Pos_{i}" for i in range(MAX_POSITIONS)],
        "Tempo": [f"Tempo_{i}" for i in range(30, 211, 5)],
        "Pitch": [f"Pitch_{i}" for i in range(21, 109)],
        "Velocity": [f"Vel_{i}" for i in range(1, 128, 4)],
        "Duration": [f"Dur_{i}" for i in range(1, MAX_POSITIONS * 2 + 1)]
    }

    vocab_list = ["<PAD>", "<BOS>", "<EOS>"] + [tok for group in VOCAB.values() for tok in group]
    tok2idx = {tok: idx for idx, tok in enumerate(vocab_list)}
    return tok2idx

def process_pop909_dataset(dataset_path, POS_PER_BEAT=16, MAX_POSITIONS=128):
    data_dir = pathlib.Path(dataset_path)
    all_song_sequences = []

    song_folders = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    print(f"Found {len(song_folders)} song folders in {dataset_path}")

    for song_folder in song_folders:
        song_name = song_folder.name
        midi_path = song_folder / f"{song_name}.mid"
        beat_path = song_folder / "beat_midi.txt"
        chord_path = song_folder / "chord_midi.txt"

        if not midi_path.exists() or not beat_path.exists() or not chord_path.exists():
            continue

        try:
            midi_data = pretty_midi.PrettyMIDI(str(midi_path))

            # Parse beats
            beat_times, beat_orders = [], []
            with open(beat_path, "r") as f:
                for line in f:
                    parts = line.split()  # [start_time(sec), strong/weak beat, beat_order(is it a downbeat?)]
                    if not parts: continue
                    beat_times.append(round(float(parts[0]), 6))
                    beat_orders.append(float(parts[2]) if len(parts) > 2 else 0.0)

            beat_info = get_beat_info(beat_orders)

            # Parse tempos
            tempo_times, tempi = midi_data.get_tempo_changes()
            tempo_events = [{"time": round(float(t), 6), "qpm": round(qpm)} for t, qpm in zip(tempo_times, tempi)]

            all_events = []

            # Extract Notes from tracks
            for inst in midi_data.instruments:  # POP909 has three tracks: MELODY, BRIDGE, PIANO
                for note in inst.notes:
                    bar, pos, b_len = time_to_grid(note.start, beat_times, beat_info)
                    dur = max(1, min(MAX_POSITIONS * 2, int(((note.end - note.start) / b_len) * POS_PER_BEAT)))
                    vel = max(1, note.velocity // 4 * 4 + 1)
                    all_events.append({"bar": bar, "pos": pos, "type": "note", "pitch": note.pitch, "vel": vel, "dur": dur})

            # Add tempo to events
            for t in tempo_events:
                bar, pos, _ = time_to_grid(t["time"], beat_times, beat_info)
                val = max(30, min(210, int(t["qpm"]) // 5 * 5))
                all_events.append({"bar": bar, "pos": pos, "type": "tempo", "val": val})

            # Group by bar
            bars = {}
            for ev in all_events:
                bars.setdefault(ev["bar"], []).append(ev)

            max_bar = max(bars.keys()) if bars else 0

            # Find the first bar containing any notes (avoiding silent intros)
            first_note_bar = None
            for b in range(max_bar + 1):
                if any(e.get("type") == "note" for e in bars.get(b, [])):
                    first_note_bar = b
                    break

            if first_note_bar is None:
                continue

            seq_rep2 = ["<BOS>"]
            for bar_idx in range(first_note_bar, max_bar + 1):
                seq_rep2.append("Bar")
                if bar_idx not in bars: continue

                type_order = {"tempo": 0, "note": 1}

                def sort_key(e):
                    # Sorts purely by Position -> Type -> Pitch (for notes)
                    return (e["pos"], type_order[e["type"]], e.get("pitch", 0))

                sorted_events = sorted(bars[bar_idx], key=sort_key)

                for e in sorted_events:
                    seq_rep2.append(f"Pos_{e['pos']}")
                    if e["type"] == "tempo":
                        seq_rep2.append(f"Tempo_{e['val']}")
                    elif e["type"] == "note":
                        seq_rep2.extend([f"Pitch_{e['pitch']}", f"Vel_{e['vel']}", f"Dur_{e['dur']}"])

            seq_rep2.append("<EOS>")

            all_song_sequences.append({
                "song_name": song_name,
                "tokens": seq_rep2
            })
            print(f"Processed {song_name} successfully - Seq length: {len(seq_rep2)}")

        except Exception as e:
            print(f"Error processing {song_name}: {e}")

    return all_song_sequences

if __name__ == "__main__":
    args = parse_args()

    POS_PER_BEAT = args.pos_per_beat
    MAX_POSITIONS = 8 * POS_PER_BEAT

    pathlib.Path(args.tok2idx_output).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.sequences_output).parent.mkdir(parents=True, exist_ok=True)

    tok2idx = create_vocabulary(args.dataset_path, MAX_POSITIONS)
    print(f"Vocabulary size: {len(tok2idx)} tokens")

    all_song_sequences = process_pop909_dataset(args.dataset_path, POS_PER_BEAT, MAX_POSITIONS)

    with open(args.tok2idx_output, "w") as f:
        json.dump(tok2idx, f, indent=4)

    with open(args.sequences_output, "w") as f:
        json.dump(all_song_sequences, f)