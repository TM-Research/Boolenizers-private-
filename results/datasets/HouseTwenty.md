# HouseTwenty

`time_series` · 2 classes · 111 train / 48 test · 512 features · imbalance 1.3×

**Winner: TM** — TM 0.915 (OnlineATRBinarizer) vs ML 0.871 (ExtraTrees), Δ +0.044


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | OnlineATRBinarizer | 0.9153 |
| 2 | AdaptiveGaussian | 0.8947 |
| 3 | OGBFast | 0.8947 |
| 4 | OnlineBollingerBinarizer | 0.8947 |
| 5 | PulseResonanceBinarizer | 0.8947 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| ExtraTrees | 0.8714 |
| LightGBM | 0.8536 |
| XGBoost | 0.8330 |
| RandomForest | 0.8322 |
