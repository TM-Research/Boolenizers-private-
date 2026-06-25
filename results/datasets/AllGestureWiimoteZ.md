# AllGestureWiimoteZ

`time_series` · 10 classes · 700 train / 300 test · 500 features · imbalance 1.0×

**Winner: ML** — TM 0.471 (NTEUniform) vs ML 0.479 (ExtraTrees), Δ -0.008


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | NTEUniform | 0.4707 |
| 2 | SignalQuantileFusion | 0.4507 |
| 3 | StandardBinarizerNative | 0.4439 |
| 4 | NTEBatchQuantile | 0.4382 |
| 5 | TWINEv2 | 0.4366 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| ExtraTrees | 0.4786 |
| RandomForest | 0.4091 |
| LightGBM | 0.3912 |
| XGBoost | 0.3735 |
