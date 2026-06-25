# FreezerRegularTrain

`time_series` · 2 classes · 2100 train / 900 test · 301 features · imbalance 1.0×

**Winner: ML** — TM 0.999 (DualDynamicsBinarizer) vs ML 1.000 (RandomForest), Δ -0.001


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | DualDynamicsBinarizer | 0.9989 |
| 2 | OGBFast | 0.9989 |
| 3 | OnlineBollingerBinarizer | 0.9989 |
| 4 | ACFB | 0.9978 |
| 5 | MovingWindowBinarizer | 0.9978 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| RandomForest | 1.0000 |
| ExtraTrees | 0.9989 |
| LightGBM | 0.9978 |
| XGBoost | 0.9967 |
