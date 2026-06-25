# cicmaldroid-2020-multiclass

`tabular` · 5 classes · 9278 train / 2320 test · 470 features · imbalance 3.1×

**Winner: TM** — TM 0.990 (OnlineRSIMACDBinarizer) vs ML 0.951 (LightGBM), Δ +0.039


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | OnlineRSIMACDBinarizer | 0.9895 |
| 2 | AdaptiveGaussian | 0.9880 |
| 3 | OnlineQuantileSignalBinarizer | 0.9862 |
| 4 | OnlineDeltaMomentumBinarizer | 0.9806 |
| 5 | OQSB | 0.9802 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| LightGBM | 0.9508 |
| XGBoost | 0.9474 |
| ExtraTrees | 0.9386 |
| RandomForest | 0.9368 |
