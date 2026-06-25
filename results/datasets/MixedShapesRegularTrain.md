# MixedShapesRegularTrain

`time_series` · 5 classes · 2047 train / 878 test · 512 features · imbalance 1.8×

**Winner: TM** — TM 0.950 (TWINEv2) vs ML 0.937 (LightGBM), Δ +0.013


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | TWINEv2 | 0.9503 |
| 2 | KalmanFilterBinarizer | 0.9494 |
| 3 | TWINELite | 0.9490 |
| 4 | SpectralStabilityBinarizer | 0.9483 |
| 5 | ACFB | 0.9477 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| LightGBM | 0.9374 |
| XGBoost | 0.9344 |
| ExtraTrees | 0.9337 |
| RandomForest | 0.9261 |
