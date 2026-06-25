# GunPointAgeSpan

`time_series` · 2 classes · 315 train / 136 test · 150 features · imbalance 1.0×

**Winner: TM** — TM 0.985 (NTEUniform) vs ML 0.963 (XGBoost), Δ +0.022


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | NTEUniform | 0.9853 |
| 2 | OnlineGeneralizedBinarizer | 0.9853 |
| 3 | AdaptiveMomentumBinarizer | 0.9779 |
| 4 | GLADEBooleanizer | 0.9779 |
| 5 | MovingWindowBinarizerV2 | 0.9779 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| XGBoost | 0.9632 |
| LightGBM | 0.9632 |
| RandomForest | 0.9632 |
| ExtraTrees | 0.9632 |
