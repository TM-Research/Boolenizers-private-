# cic-iov-2024-multiclass

`tabular` · 6 classes · 30000 train / 10000 test · 8 features · imbalance 122.4×

**Winner: TM** — TM 0.979 (TWINEv2) vs ML 0.975 (XGBoost), Δ +0.004


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | TWINEv2 | 0.9789 |
| 2 | AQB | 0.9751 |
| 3 | DriftRobustBinarizer | 0.9751 |
| 4 | DualDynamicsBinarizer | 0.9751 |
| 5 | GLADEBooleanizer | 0.9751 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| XGBoost | 0.9751 |
| LightGBM | 0.9732 |
| RandomForest | 0.9732 |
| ExtraTrees | 0.9732 |
