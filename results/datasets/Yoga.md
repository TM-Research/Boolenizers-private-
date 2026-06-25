# Yoga

`time_series` · 2 classes · 2310 train / 990 test · 426 features · imbalance 1.2×

**Winner: ML** — TM 0.934 (OnlineGeneralizedBinarizer) vs ML 0.946 (ExtraTrees), Δ -0.012


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | OnlineGeneralizedBinarizer | 0.9340 |
| 2 | StandardBinarizerNative | 0.9309 |
| 3 | OGBFast | 0.9258 |
| 4 | SAQT | 0.9257 |
| 5 | GLADEBooleanizer | 0.9239 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| ExtraTrees | 0.9460 |
| LightGBM | 0.9410 |
| RandomForest | 0.9389 |
| XGBoost | 0.9319 |
