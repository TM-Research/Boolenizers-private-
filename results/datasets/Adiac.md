# Adiac

`time_series` · 37 classes · 546 train / 235 test · 176 features · imbalance 1.4×

**Winner: TM** — TM 0.765 (GLADEEncoder) vs ML 0.702 (RandomForest), Δ +0.064


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | GLADEEncoder | 0.7653 |
| 2 | QBEspresso | 0.7604 |
| 3 | GLADEBooleanizer | 0.7601 |
| 4 | StandardBinarizerNative | 0.7563 |
| 5 | SAQT | 0.7562 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| RandomForest | 0.7016 |
| ExtraTrees | 0.6921 |
| XGBoost | 0.6706 |
| LightGBM | 0.6663 |
