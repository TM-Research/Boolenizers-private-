# PowerCons

`time_series` · 2 classes · 252 train / 108 test · 144 features · imbalance 1.0×

**Winner: TM** — TM 0.972 (AdaptiveGaussian) vs ML 0.963 (XGBoost), Δ +0.009


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | AdaptiveGaussian | 0.9722 |
| 2 | NTEUniform | 0.9722 |
| 3 | OnlineQuantileTrackerBinarizer | 0.9722 |
| 4 | SAQT | 0.9722 |
| 5 | SSL | 0.9722 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| XGBoost | 0.9630 |
| LightGBM | 0.9630 |
| RandomForest | 0.9629 |
| ExtraTrees | 0.9629 |
