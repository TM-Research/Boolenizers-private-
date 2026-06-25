# DodgerLoopGame

`time_series` · 2 classes · 100 train / 44 test · 288 features · imbalance 1.0×

**Winner: ML** — TM 0.885 (TWINEv2) vs ML 0.886 (ExtraTrees), Δ -0.001


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | TWINEv2 | 0.8849 |
| 2 | OnlineATRBinarizer | 0.8634 |
| 3 | ACFB | 0.8625 |
| 4 | SpectralStabilityBinarizer | 0.8625 |
| 5 | DriftRobustBinarizer | 0.8408 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| ExtraTrees | 0.8863 |
| RandomForest | 0.8408 |
| LightGBM | 0.7953 |
| XGBoost | 0.7267 |
