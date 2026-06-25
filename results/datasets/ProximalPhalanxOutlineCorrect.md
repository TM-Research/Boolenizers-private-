# ProximalPhalanxOutlineCorrect

`time_series` · 2 classes · 623 train / 268 test · 80 features · imbalance 2.1×

**Winner: TM** — TM 0.851 (SAQT) vs ML 0.841 (XGBoost), Δ +0.010


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | SAQT | 0.8506 |
| 2 | TWINEv2 | 0.8491 |
| 3 | ACFB | 0.8446 |
| 4 | OnlineQuantileSignalBinarizer | 0.8429 |
| 5 | AdaptiveQuantileBinarizer | 0.8411 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| XGBoost | 0.8408 |
| RandomForest | 0.8218 |
| ExtraTrees | 0.8216 |
| LightGBM | 0.8142 |
