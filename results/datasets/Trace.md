# Trace

`time_series` · 4 classes · 140 train / 60 test · 275 features · imbalance 1.0×

**Winner: TM** — TM 1.000 (DecisionTreeBinarizer) vs ML 0.967 (ExtraTrees), Δ +0.033


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | DecisionTreeBinarizer | 1.0000 |
| 2 | DriftRobustBinarizer | 1.0000 |
| 3 | KalmanFilterBinarizer | 1.0000 |
| 4 | MovingWindowBinarizer | 1.0000 |
| 5 | NTEUniform | 1.0000 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| ExtraTrees | 0.9667 |
| LightGBM | 0.9499 |
| RandomForest | 0.9330 |
| XGBoost | 0.8991 |
