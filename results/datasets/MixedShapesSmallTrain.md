# MixedShapesSmallTrain

`time_series` · 5 classes · 1767 train / 758 test · 512 features · imbalance 2.0×

**Winner: TM** — TM 0.959 (TWINEv3) vs ML 0.952 (XGBoost), Δ +0.007


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | TWINEv3 | 0.9589 |
| 2 | KalmanFilterBinarizer | 0.9574 |
| 3 | OnlineBollingerBinarizer | 0.9572 |
| 4 | TWINELite | 0.9570 |
| 5 | ResonantGradientBinarizer | 0.9567 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| XGBoost | 0.9521 |
| LightGBM | 0.9437 |
| ExtraTrees | 0.9416 |
| RandomForest | 0.9276 |
