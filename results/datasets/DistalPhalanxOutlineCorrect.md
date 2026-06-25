# DistalPhalanxOutlineCorrect

`time_series` · 2 classes · 613 train / 263 test · 80 features · imbalance 1.6×

**Winner: TM** — TM 0.856 (SketchTDigest) vs ML 0.849 (LightGBM), Δ +0.008


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | SketchTDigest | 0.8564 |
| 2 | GLADEBooleanizer | 0.8505 |
| 3 | SignalGradientBinarizer | 0.8505 |
| 4 | NTEBatchQuantile | 0.8430 |
| 5 | DriftRobustBinarizer | 0.8424 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| LightGBM | 0.8486 |
| RandomForest | 0.8461 |
| ExtraTrees | 0.8461 |
| XGBoost | 0.8133 |
