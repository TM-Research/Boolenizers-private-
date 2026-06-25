# domain-info-2024-multiclass

`tabular` · 3 classes · 29999 train / 10000 test · 26 features · imbalance 8.2×

**Winner: TM** — TM 1.000 (GLADEBooleanizer) vs ML 1.000 (XGBoost), Δ +0.000


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | GLADEBooleanizer | 1.0000 |
| 2 | GLADEEncoder | 1.0000 |
| 3 | MovingWindowBinarizerV2 | 1.0000 |
| 4 | OQSB | 1.0000 |
| 5 | OnlineGeneralizedBinarizer | 1.0000 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| XGBoost | 1.0000 |
| LightGBM | 1.0000 |
| ExtraTrees | 0.9962 |
| RandomForest | 0.9899 |
