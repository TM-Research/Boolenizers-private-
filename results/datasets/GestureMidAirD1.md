# GestureMidAirD1

`time_series` · 26 classes · 236 train / 102 test · 360 features · imbalance 1.1×

**Winner: TM** — TM 0.601 (NTEUniform) vs ML 0.575 (RandomForest), Δ +0.026


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | NTEUniform | 0.6012 |
| 2 | AdaptiveQuantileBinarizer | 0.5803 |
| 3 | DriftRobustBinarizer | 0.5723 |
| 4 | QBEspresso | 0.5658 |
| 5 | StandardBinarizerNative | 0.5610 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| RandomForest | 0.5751 |
| LightGBM | 0.5594 |
| ExtraTrees | 0.5489 |
| XGBoost | 0.5030 |
