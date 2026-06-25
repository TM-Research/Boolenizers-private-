# InsectEPGRegularTrain

`time_series` · 3 classes · 217 train / 94 test · 512 features · imbalance 2.9×

**Winner: TM** — TM 0.760 (TWINELite) vs ML 0.734 (ExtraTrees), Δ +0.026


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | TWINELite | 0.7596 |
| 2 | QBEspresso | 0.7565 |
| 3 | OnlineQuantileTrackerBinarizer | 0.7513 |
| 4 | SAQT | 0.7513 |
| 5 | SingleSpeedP2 | 0.7512 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| ExtraTrees | 0.7335 |
| LightGBM | 0.7002 |
| XGBoost | 0.6689 |
| RandomForest | 0.6043 |
