# Ham

`time_series` · 2 classes · 149 train / 65 test · 431 features · imbalance 1.1×

**Winner: TM** — TM 0.954 (SDQB) vs ML 0.922 (LightGBM), Δ +0.031


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | SDQB | 0.9537 |
| 2 | TWINEv2 | 0.9537 |
| 3 | AdaptiveGaussian | 0.9383 |
| 4 | SingleSpeedP2 | 0.9383 |
| 5 | AdaptiveQuantileBinarizer | 0.9381 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| LightGBM | 0.9224 |
| RandomForest | 0.8922 |
| XGBoost | 0.8603 |
| ExtraTrees | 0.8452 |
