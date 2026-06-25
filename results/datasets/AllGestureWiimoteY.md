# AllGestureWiimoteY

`time_series` · 10 classes · 700 train / 300 test · 500 features · imbalance 1.0×

**Winner: ML** — TM 0.502 (TWINEv3) vs ML 0.610 (ExtraTrees), Δ -0.107


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | TWINEv3 | 0.5025 |
| 2 | NTEUniform | 0.4807 |
| 3 | OnlineATRBinarizer | 0.4721 |
| 4 | ACFB | 0.4702 |
| 5 | NTEBatchQuantile | 0.4672 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| ExtraTrees | 0.6095 |
| RandomForest | 0.5083 |
| LightGBM | 0.4323 |
| XGBoost | 0.4306 |
