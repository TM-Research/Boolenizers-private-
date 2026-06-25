# tinyml-cs

`tabular` · 3 classes · 29999 train / 10000 test · 16 features · imbalance 5.0×

**Winner: ML** — TM 0.672 (SAQT) vs ML 0.821 (LightGBM), Δ -0.148


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | SAQT | 0.6721 |
| 2 | OnlineGeneralizedBinarizer | 0.6448 |
| 3 | AQB | 0.6351 |
| 4 | NTEBatchQuantile | 0.6228 |
| 5 | GLADEBooleanizer | 0.6156 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| LightGBM | 0.8205 |
| XGBoost | 0.8194 |
| RandomForest | 0.7336 |
| ExtraTrees | 0.4667 |
