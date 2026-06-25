# AllGestureWiimoteX

`time_series` · 10 classes · 700 train / 300 test · 500 features · imbalance 1.0×

**Winner: ML** — TM 0.405 (ACFB) vs ML 0.460 (ExtraTrees), Δ -0.055


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | ACFB | 0.4050 |
| 2 | NTEUniform | 0.3878 |
| 3 | StandardBinarizerWrapper | 0.3869 |
| 4 | OnlineUniversalBinarizer | 0.3868 |
| 5 | TWINEv3 | 0.3838 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| ExtraTrees | 0.4603 |
| RandomForest | 0.4048 |
| LightGBM | 0.3332 |
| XGBoost | 0.3197 |
