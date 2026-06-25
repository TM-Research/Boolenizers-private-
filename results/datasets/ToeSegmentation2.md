# ToeSegmentation2

`time_series` · 2 classes · 116 train / 50 test · 343 features · imbalance 3.0×

**Winner: TM** — TM 0.812 (OnlineBollingerBinarizer) vs ML 0.767 (LightGBM), Δ +0.046


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | OnlineBollingerBinarizer | 0.8125 |
| 2 | QBEspresso | 0.8125 |
| 3 | ACFB | 0.7890 |
| 4 | AdaptiveGaussian | 0.7890 |
| 5 | MovingWindowBinarizer | 0.7890 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| LightGBM | 0.7669 |
| ExtraTrees | 0.7290 |
| XGBoost | 0.6875 |
| RandomForest | 0.6811 |
