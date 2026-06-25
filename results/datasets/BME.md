# BME

`time_series` · 3 classes · 126 train / 54 test · 128 features · imbalance 1.0×

**Winner: TM** — TM 1.000 (ACFB) vs ML 1.000 (RandomForest), Δ +0.000


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | ACFB | 1.0000 |
| 2 | AdaptiveGaussian | 1.0000 |
| 3 | DynamicPulseBinarizer | 1.0000 |
| 4 | GLADEEncoder | 1.0000 |
| 5 | KalmanFilterBinarizer | 1.0000 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| RandomForest | 1.0000 |
| ExtraTrees | 1.0000 |
| XGBoost | 0.9444 |
| LightGBM | 0.9444 |
