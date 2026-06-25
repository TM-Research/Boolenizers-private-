# Earthquakes

`time_series` · 2 classes · 322 train / 139 test · 512 features · imbalance 4.0×

**Winner: TM** — TM 0.568 (MWAB) vs ML 0.557 (LightGBM), Δ +0.011


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | MWAB | 0.5675 |
| 2 | SketchTDigest | 0.5519 |
| 3 | SSL | 0.5360 |
| 4 | OnlineDeltaMomentumBinarizer | 0.5358 |
| 5 | KalmanFilterBinarizer | 0.5328 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| LightGBM | 0.5566 |
| XGBoost | 0.4910 |
| RandomForest | 0.4440 |
| ExtraTrees | 0.4440 |
