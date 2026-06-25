# DistalPhalanxOutlineAgeGroup

`time_series` · 3 classes · 377 train / 162 test · 80 features · imbalance 7.0×

**Winner: ML** — TM 0.842 (ResonantGradientBinarizerV2) vs ML 0.844 (ExtraTrees), Δ -0.002


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | ResonantGradientBinarizerV2 | 0.8419 |
| 2 | ACFB | 0.8242 |
| 3 | PulseResonanceBinarizer | 0.8234 |
| 4 | SignalQuantileFusion | 0.8148 |
| 5 | KnownMethodsBinarizer | 0.8136 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| ExtraTrees | 0.8438 |
| RandomForest | 0.8227 |
| LightGBM | 0.8056 |
| XGBoost | 0.7752 |
