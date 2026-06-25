# SonyAIBORobotSurface2

`time_series` · 2 classes · 686 train / 294 test · 65 features · imbalance 1.6×

**Winner: TM** — TM 0.972 (SingleSpeedP2) vs ML 0.961 (LightGBM), Δ +0.011


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | SingleSpeedP2 | 0.9716 |
| 2 | TWINEv3 | 0.9681 |
| 3 | KalmanFilterBinarizer | 0.9646 |
| 4 | QBEspresso | 0.9646 |
| 5 | SignalQuantileFusion | 0.9646 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| LightGBM | 0.9609 |
| XGBoost | 0.9539 |
| ExtraTrees | 0.9363 |
| RandomForest | 0.9330 |
