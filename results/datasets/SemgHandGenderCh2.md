# SemgHandGenderCh2

`time_series` · 2 classes · 630 train / 270 test · 512 features · imbalance 1.5×

**Winner: TM** — TM 0.839 (TWINEv2) vs ML 0.807 (LightGBM), Δ +0.032


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | TWINEv2 | 0.8391 |
| 2 | ACFB | 0.8314 |
| 3 | DriftRobustBinarizer | 0.8278 |
| 4 | OnlineGeneralizedBinarizer | 0.8266 |
| 5 | NTEUniform | 0.8232 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| LightGBM | 0.8074 |
| RandomForest | 0.7905 |
| ExtraTrees | 0.7845 |
| XGBoost | 0.7829 |
