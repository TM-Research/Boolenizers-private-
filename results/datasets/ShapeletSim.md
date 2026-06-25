# ShapeletSim

`time_series` · 2 classes · 140 train / 60 test · 500 features · imbalance 1.0×

**Winner: TM** — TM 0.569 (SSL) vs ML 0.483 (LightGBM), Δ +0.086


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | SSL | 0.5694 |
| 2 | TWINE | 0.5623 |
| 3 | DecisionTreeBinarizer | 0.5438 |
| 4 | MWAB | 0.5438 |
| 5 | TWINEv2 | 0.5417 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| LightGBM | 0.4832 |
| XGBoost | 0.4498 |
| ExtraTrees | 0.4498 |
| RandomForest | 0.4231 |
