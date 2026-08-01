# Predicting Fish Diversity From Passive Acoustic Monitoring

Passive acoustic monitoring offers a low-cost alternative to physical sampling for tracking fish
communities, but the relationship between recorded sound and measured diversity is not well
established. This project pairs underwater recordings with concurrent survey counts and evaluates
whether learned representations of the acoustic signal predict rarefied Shannon diversity.

Convolutional and recurrent networks are trained on spectrograms. Out-of-fold predictions from the
best sequence model are then supplied as a feature to gradient-boosted tree and random forest models
alongside environmental covariates and hand-crafted acoustic indices.

## Data

| file | contents |
|---|---|
| `Final_Fish_counts.csv` | species counts per recording, indexed `YYYY-MM-DD Day/Night` |
| `Temp_Wind_Data.csv` | sea surface temperature, wind speed and direction per session |
| `audio/*.wav` | recordings, one per session (not tracked) |

72 recordings across 42 dates, sampled approximately weekly over one year, cut into 2,670
non-overlapping 10-second clips.

## Notebooks

Run in order.

| notebook | purpose | key output |
|---|---|---|
| `01_download_audio` | fetch recordings, verify a common sample rate | `audio/` |
| `02_exploratory_analysis` | amplitude and zero-crossing summaries by season, time of day, temperature and wind | `figures/eda/` |
| `03_make_spectrograms` | spectrograms and the clip-level feature table | `merged_features.csv` |
| `04_train_cnn` | CNN, ResNet and DenseNet | `runs/`, `figures/training/` |
| `05_train_rnn` | bidirectional LSTM and GRU | `runs/`, `figures/training/` |
| `06_results_summary` | comparison across architectures and seeds | `figures/results/` |
| `07_gru_predictions` | GRU predictions and residual autocorrelation | `merged_features_with_pred.csv`, `figures/gru/` |
| `08_tree_models` | gradient-boosted trees and random forests, feature-set ablation | `figures/trees/` |

## Method

**Spectrograms.** Recordings are normalised by the 99th percentile of their raw amplitude,
band-passed to 100–2000 Hz, resampled to 6 kHz and cut into 10-second clips. Each clip is
transformed with a 1024-point Hann window and 256-sample hop, giving a 324 × 235 representation.

**Features.** Acoustic measurements are made on the 100–2000 Hz band, the range in which fish sound
is concentrated. One exception carries a `_fullband` suffix: zero-crossing rate counts sign changes
per sample, so band-limiting would make it a property of the filter rather than of the sound.

**Targets.** Rarefied Shannon diversity and rarefied species richness, computed on the species counts
with *Anchoa* excluded and standardised to a common sampling depth of 122 individuals. *Anchoa* is a
schooling species that dominated individual catches, so a single sample could swamp any diversity
measure computed from raw counts.

**Evaluation.** Five-fold cross-validation split by recording, so no clip from a test recording
appears in training. Fold assignments are shared across all models via `runs/folds_k5_seed*.json`,
and every configuration is repeated over five seeds. Pooled R² over all held-out recordings is the
headline metric; MAE and RMSE are reported per fold. Baselines predict the training mean and median.

**Integration.** Notebook 08 aggregates the clip-level table to one row per recording, since the
target is a property of the whole recording. Species counts and total abundance are excluded, as both
derive from the survey the target is computed from. Six feature sets isolate what each source
contributes; three of them differ from a counterpart only by the presence of the GRU prediction.

## Results

Mean over five seeds, with the seed-to-seed standard deviation.

**Neural networks, Shannon diversity** (mean-baseline MAE 0.3073)

| model | R² | MAE |
|---|---|---|
| BiLSTM | 0.183 ± 0.065 | 0.271 |
| BiGRU | 0.166 ± 0.047 | 0.276 |
| DenseNet | 0.127 ± 0.093 | 0.282 |
| CNN | 0.086 ± 0.037 | 0.287 |
| ResNet | 0.080 ± 0.044 | 0.286 |

Sequence models outperform image models on both diversity targets.

**Tree models by feature set, random forest** (mean-baseline MAE 0.3005)

| feature set | features | R² | MAE |
|---|---|---|---|
| environment + GRU | 8 | 0.329 ± 0.032 | 0.236 |
| environment | 7 | 0.295 ± 0.030 | 0.246 |
| environment + acoustic + GRU | 45 | 0.203 ± 0.048 | 0.264 |
| environment + acoustic | 44 | 0.181 ± 0.036 | 0.269 |
| acoustic + GRU | 38 | 0.167 ± 0.051 | 0.273 |
| acoustic | 37 | 0.148 ± 0.043 | 0.276 |

Adding the GRU prediction improves every background it is added to, in 15 of 15 paired runs for the
random forest. The hand-crafted
acoustic block reduces performance in every combination it appears in, consistent with 37 correlated
features exceeding what 72 recordings support. XGBoost agrees on two of the three paired comparisons.

Effect sizes are modest relative to the seasonal signal carried by sea temperature and time of year.

## Outputs

- `merged_features.csv` — one row per clip: acoustic features, environmental readings, species counts
  and diversity targets (116 columns)
- `merged_features_with_pred.csv` — the same, plus `shannon_pred_gru`, the out-of-fold GRU prediction
  averaged over five seeds (117 columns)

Figures are written under `figures/`.

## Layout

```
.
├── 01_download_audio.ipynb  …  08_tree_models.ipynb
├── Final_Fish_counts.csv
├── Temp_Wind_Data.csv
├── merged_features.csv
├── merged_features_with_pred.csv
├── figures/             eda, training, results, gru, trees
├── audio/               recordings (not tracked)
├── prepared/            spectrogram cache (not tracked)
└── runs/                per-run results and fold assignments (not tracked)
```


## Requirements

Python 3.11 or later.

```
pip install -r requirements.txt
```

