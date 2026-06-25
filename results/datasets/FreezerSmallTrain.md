# FreezerSmallTrain

`time_series` · 2 classes · 2014 train / 864 test · 301 features · imbalance 1.0×

**Winner: TM** — TM 1.000 (OGBFast) vs ML 1.000 (LightGBM), Δ +0.000


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | OGBFast | 1.0000 |
| 2 | OnlineBollingerBinarizer | 1.0000 |
| 3 | ACFB | 0.9988 |
| 4 | MovingWindowBinarizer | 0.9988 |
| 5 | OnlineUniversalBinarizer | 0.9988 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| LightGBM | 1.0000 |
| RandomForest | 1.0000 |
| ExtraTrees | 0.9988 |
| XGBoost | 0.9965 |
