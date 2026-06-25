# Coffee

`time_series` · 2 classes · 39 train / 17 test · 286 features · imbalance 1.1×

**Winner: TM** — TM 1.000 (AdaptiveGaussian) vs ML 0.941 (XGBoost), Δ +0.059


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | AdaptiveGaussian | 1.0000 |
| 2 | AdaptiveMomentumBinarizer | 1.0000 |
| 3 | KBinsThermometer | 1.0000 |
| 4 | MovingWindowBinarizer | 1.0000 |
| 5 | OnlineRSIMACDBinarizer | 1.0000 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| XGBoost | 0.9412 |
| RandomForest | 0.9412 |
| ExtraTrees | 0.9412 |
| LightGBM | 0.3462 |
