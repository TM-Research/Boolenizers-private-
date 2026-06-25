# Plane

`time_series` · 7 classes · 147 train / 63 test · 144 features · imbalance 1.0×

**Winner: TM** — TM 1.000 (AdaptiveGaussian) vs ML 0.984 (LightGBM), Δ +0.016


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | AdaptiveGaussian | 1.0000 |
| 2 | OnlineRSIMACDBinarizer | 1.0000 |
| 3 | QBEspresso | 1.0000 |
| 4 | SingleSpeedP2 | 1.0000 |
| 5 | ACFB | 0.9841 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| LightGBM | 0.9841 |
| RandomForest | 0.9841 |
| ExtraTrees | 0.9841 |
| XGBoost | 0.9519 |
