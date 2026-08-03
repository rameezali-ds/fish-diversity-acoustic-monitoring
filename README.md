# Predicting Fish Diversity From Passive Acoustic Monitoring

Passive acoustic monitoring offers a low-cost alternative to physical sampling for tracking fish
communities, but the relationship between recorded sound and measured diversity is not well
established. This project pairs underwater recordings with concurrent survey counts and asks a
single question: can the acoustic signal predict rarefied Shannon diversity?

Convolutional and recurrent networks are trained on spectrograms. Out-of-fold predictions from the
best sequence model are then supplied as a feature to gradient-boosted tree and random forest models
alongside hand-crafted acoustic indices, which tests whether a learned representation of the
spectrogram carries information those summaries miss.

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
| `02_exploratory_analysis` | amplitude and zero-crossing summaries by season and time of day | `figures/eda/` |
| `03_make_spectrograms` | spectrograms and the clip-level feature table | `merged_features.csv` |
| `04_train_cnn` | CNN, ResNet and DenseNet | `runs/`, `figures/training/` |
| `05_train_rnn` | bidirectional LSTM and GRU | `runs/`, `figures/training/` |
| `06_results_summary` | comparison across architectures and seeds | `figures/results/` |
| `07_gru_predictions` | GRU predictions and residual autocorrelation | `merged_features_with_pred.csv`, `figures/gru/` |
| `08_tree_models` | tree models on acoustic features, with and without the GRU prediction | `figures/trees/` |

## Method

**Spectrograms.** Recordings are normalised by the 99th percentile of their raw amplitude,
band-passed to 100–2000 Hz, resampled to 6 kHz and cut into 10-second clips. Each clip is
transformed with a 1024-point Hann window and 256-sample hop, giving a 324 × 235 representation.
A sweep over window lengths from 32 ms to 341 ms found results insensitive to this choice.

**Features.** Acoustic measurements are made on the 100–2000 Hz band, the range in which fish sound
is concentrated. One exception carries a `_fullband` suffix: zero-crossing rate counts sign changes
per sample, so band-limiting would make it a property of the filter rather than of the sound.

**Targets.** Rarefied Shannon diversity and rarefied species richness, computed on the species counts
with *Anchoa* excluded and standardised to a common sampling depth of 122 individuals. *Anchoa* is a
schooling species that dominated individual catches, so a single sample could swamp any diversity
measure computed from raw counts. Richness uses the Hurlbert (1971) closed form; Shannon is estimated
by Monte Carlo over multivariate hypergeometric draws.

**Evaluation.** Five-fold cross-validation split by recording, so no clip from a test recording
appears in training. Fold assignments are shared across all models via `runs/folds_k5_seed*.json`,
and every configuration is repeated over five seeds. Pooled R² over all held-out recordings is the
headline metric; MAE and RMSE are reported per fold. Baselines predict the training mean and median.

**Integration.** Notebook 08 aggregates the clip-level table to one row per recording, since the
target is a property of the whole recording. Species counts and total abundance are excluded, as both
derive from the survey the target is computed from. Four feature sets are formed from two acoustic
blocks, each with and without the GRU prediction:

- **acoustic** — all 37 acoustic summaries
- **acoustic (reduced)** — RMS, zero-crossing rate and the three band energy fractions

The reduced block drops the nineteen 100 Hz power bins, which are near-linear combinations of the
band fractions, so it covers the same ground with far less redundancy.

## Results

Mean over five seeds, with the seed-to-seed standard deviation.

**Neural networks, Shannon diversity** (mean-baseline MAE 0.3073)

| model | parameters | R² | MAE |
|---|---|---|---|
| BiLSTM | 224,898 | 0.183 ± 0.065 | 0.271 |
| BiGRU | 174,978 | 0.166 ± 0.047 | 0.276 |
| DenseNet | 178,045 | 0.127 ± 0.093 | 0.282 |
| CNN | 40,161 | 0.086 ± 0.037 | 0.287 |
| ResNet | 191,601 | 0.080 ± 0.044 | 0.286 |

Sequence models outperform image models on both diversity targets. The BiLSTM and BiGRU are close
enough that the BiGRU is carried forward on the grounds of simplicity: it uses fewer gates and a
smaller internal state, so there is less capacity to overfit 72 recordings.

**Tree models on acoustic features** (mean-baseline MAE 0.3005)

| model | feature set | features | R² | MAE | RMSE |
|---|---|---|---|---|---|
| Random forest | acoustic | 37 | 0.148 ± 0.043 | 0.276 ± 0.008 | 0.343 |
| Random forest | acoustic + GRU | 38 | 0.167 ± 0.051 | 0.273 ± 0.009 | 0.339 |
| Random forest | acoustic (reduced) | 5 | 0.118 ± 0.047 | 0.283 ± 0.012 | 0.349 |
| Random forest | acoustic (reduced) + GRU | 6 | 0.238 ± 0.030 | 0.259 ± 0.008 | 0.324 |
| XGBoost | acoustic | 37 | 0.035 ± 0.089 | 0.299 ± 0.013 | 0.365 |
| XGBoost | acoustic + GRU | 38 | 0.070 ± 0.075 | 0.294 ± 0.011 | 0.358 |
| XGBoost | acoustic (reduced) | 5 | −0.040 ± 0.090 | 0.302 ± 0.016 | 0.379 |
| XGBoost | acoustic (reduced) + GRU | 6 | 0.136 ± 0.043 | 0.270 ± 0.008 | 0.345 |

Adding the GRU prediction improves R², MAE and RMSE in all four pairs. Since every set is built from
the same audio, this indicates that a network trained on the spectrogram extracts information the
hand-crafted acoustic summaries do not.

The size of that improvement depends on how many features sit alongside it. With 37 acoustic columns
the random forest gains 0.019 R²; with 5 it gains 0.120. A random forest considers only a subset of
features at each split, so a single informative column is offered far more often when the candidate
pool is small. The same pattern appears in XGBoost.

Both blocks are reported rather than one being selected. Choosing between them on the strength of
these numbers would mean picking a feature set using the same cross-validation scores then quoted as
the result, which inflates the estimate.

Effect sizes are small throughout. With 72 recordings the seed-to-seed standard deviation is
comparable to the improvement itself, particularly for XGBoost, whose spread is roughly twice the
random forest's.

## Outputs

- `merged_features.csv` — one row per clip: acoustic features, environmental readings, species counts
  and diversity targets (116 columns)
- `merged_features_with_pred.csv` — the same, plus `shannon_pred_gru`, the out-of-fold GRU prediction
  averaged over five seeds (117 columns)

`shannon_pred_gru` is out-of-fold, so it can be combined with other features without leakage provided
any downstream model is evaluated on the same folds in `runs/folds_k5_seed*.json`.

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

Paths default to `/scratch/$USER/capstone`. Change `BASE` at the top of each notebook to run
elsewhere.

## Requirements

Python 3.11 or later.

```
pip install -r requirements.txt
```

A GPU is not required but reduces training time substantially. Notebook 03 holds the spectrogram
array in memory, requiring approximately 2 GB of RAM.

## Reference

Hurlbert, S.H. (1971). The nonconcept of species diversity: a critique and alternative parameters.
*Ecology* 52(4), 577–586.
