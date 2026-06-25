# StarLightCurves

`time_series` · 3 classes · 6465 train / 2771 test · 512 features · imbalance 4.0×

**Winner: ML** — TM 0.957 (OnlineGeneralizedBinarizer) vs ML 0.960 (ExtraTrees), Δ -0.003


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | OnlineGeneralizedBinarizer | 0.9565 |
| 2 | TWINELite | 0.9565 |
| 3 | StandardBinarizerNative | 0.9554 |
| 4 | OnlineQuantileTrackerBinarizer | 0.9541 |
| 5 | SketchGK | 0.9541 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| ExtraTrees | 0.9598 |
| XGBoost | 0.9550 |
| LightGBM | 0.9532 |
| RandomForest | 0.9473 |
