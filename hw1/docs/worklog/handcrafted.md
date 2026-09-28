# Hand-crafted baseline

Done in the main session (no worker), 2026-09-28.

## Changes

- `src/hw1/features/handcrafted.py`: `extract(y) -> dict`, 97 named features in 5 groups:
  - timbre: MFCC 20 mean/std, spectral contrast 7 mean/std;
  - spectral: centroid, bandwidth, rolloff, flatness, zero-crossing rate;
  - energy: RMS dB, dynamic range, 7 band-energy ratios, spectral flux;
  - rhythm: onset strength, tempo, beat-interval CV, pulse clarity;
  - harmony: key-normalized chroma, chroma entropy, chroma change.
- `src/hw1/probe.py`: `load_features`, `split_xy`, `make_model`, `fit_eval`. Classifiers: logreg, svm_rbf (calibrated), rf, knn, each with a StandardScaler fitted on train. Reusable for MERT and CLAP.
- `scripts/extract_handcrafted.py`: features for all clips into `features/{A,B}_handcrafted.npz` (6 processes; A 128 s, B 98 s).
- `scripts/bench_handcrafted.py`: 4 models, 5 group ablations, confusion matrices, error structure, ANOVA ranking, class trend plots. Writes `results/handcrafted.json`.
- `scripts/plot_spectra.py`: per-class long-term average spectra and one example log-mel spectrogram per class (train split).
- `metrics.plot_confusion`: the title wraps so it isn't cut off.

## Decisions

- **Fixed default hyperparameters, no grid search.** The validation sets are small (132 and 102 clips), so a grid would overfit them. The model is chosen among the 4 classifiers by validation S only.
- **Chroma is rolled to the strongest pitch class** (key-normalized), because a song's key says nothing about its era or market.
- **SVM probabilities come from `CalibratedClassifierCV(SVC, ensemble=False)`.** `SVC(probability=True)` is deprecated in scikit-learn 1.9.

## Results (validation; chance: top-1 0.167, top-3 0.5, S 0.417)

| Dataset | Model | Top-1 | Top-3 | S |
|---|---|---:|---:|---:|
| A | logreg (best) | 0.364 | 0.856 | 0.792 |
| A | svm_rbf | 0.371 | 0.818 | 0.780 |
| A | rf | 0.364 | 0.788 | 0.758 |
| A | knn | 0.311 | 0.712 | 0.667 |
| B | svm_rbf (best) | 0.314 | 0.735 | 0.681 |
| B | rf | 0.324 | 0.676 | 0.662 |
| B | knn | 0.284 | 0.696 | 0.632 |
| B | logreg | 0.294 | 0.647 | 0.618 |

Each feature group alone (logreg), S on validation:

| Group | A | B |
|---|---:|---:|
| energy (12) | 0.761 | 0.613 |
| timbre (54) | 0.720 | 0.627 |
| spectral (10) | 0.572 | 0.485 |
| harmony (16) | 0.572 | 0.559 |
| rhythm (5) | 0.508 | 0.466 |

Figures in `results/`:
- `handcrafted_{A,B}_cm.png`, `handcrafted_{A,B}_cm_norm.png`: confusion matrices, counts and row-normalized.
- `handcrafted_{A,B}_trends.png`: class-mean trends of key features.
- `spectrum_{A,B}_ltas.png`: average spectrum per class.
- `spectrum_{A,B}_examples.png`: one example spectrogram per class.

## Findings

- **A: production cues.**
  - Loudness: mean RMS is flat through the 1980s (about 43 dB), then rises to 46.5 dB in the 2010s, while dynamic range falls from 12.4 to 10.7 dB.
  - Sub-bass: the share of energy below 60 Hz rises steadily, from 2.3% (1960s) to 8.5% (2010s).
  - Brightness: it peaks in the 1980s-90s, with centroid and flatness highest there. The 1960s-70s are 2-3.5 dB darker above 5 kHz.
- **A: errors.** 47.6% of validation errors are to a neighboring decade; random wrong guesses would give 33%. The main confusion blocks are 2000s with 2010s, and 1960s with 1970s.
- **B: collapse onto UK.** The best model never predicts US. 9 of 17 US clips go to UK, and Spain and Germany are mostly sent to UK or Brazil. The class spectra differ by only about ±2 dB, so production-level features carry little market information.

## Tests

`uv run pytest`: 24 passed.
- `tests/test_handcrafted.py`, synthetic signals: sine centroid and band, noise flatness, a 20 dB level step, click-track tempo.
- `tests/test_probe.py`: separable data is learned by all 4 classifiers.

## Open issues

- A: the top-3 of 0.856 is already high. Ordinal smoothing may add a little more.
- B needs language and style cues (MERT, Whisper). US recall is 0 with hand-crafted features.

## Combination and mutual information check (`scripts/analyze_mi.py`, `results/handcrafted_mi.json`)

- **The groups are not redundant with energy:** mean |correlation| is 0.12-0.26. Timbre and harmony carry mutual information comparable to energy.
- **Adding groups to energy barely helps** (logreg, validation S):
  - A: energy 0.761, +timbre 0.811, +spectral 0.777, +rhythm 0.765, +harmony 0.724, all 0.792.
  - B: energy 0.613, +timbre 0.632, all 0.618.
- **Top-k by MI is worse than using all features** (A top-20: 0.742), because univariate MI misses features that only help in combination.
- **Validation noise:** with 132 validation clips, S has roughly ±0.05 sampling noise, so these differences aren't significant. Don't pick a subset by validation (that would be tuning on validation). Keep all features; get gains from new information sources (MERT, language ID) through late fusion.
