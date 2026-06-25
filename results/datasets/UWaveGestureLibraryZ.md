# UWaveGestureLibraryZ

`time_series` · 8 classes · 3134 train / 1344 test · 315 features · imbalance 1.0×

**Winner: TM** — TM 0.762 (ACFB) vs ML 0.760 (ExtraTrees), Δ +0.002


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | ACFB | 0.7620 |
| 2 | NTEUniform | 0.7575 |
| 3 | SketchGK | 0.7574 |
| 4 | PulseResonanceBinarizer | 0.7569 |
| 5 | SketchTDigest | 0.7564 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| ExtraTrees | 0.7599 |
| XGBoost | 0.7527 |
| LightGBM | 0.7498 |
| RandomForest | 0.7468 |
