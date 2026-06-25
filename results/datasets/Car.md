# Car

`time_series` · 4 classes · 84 train / 36 test · 512 features · imbalance 1.0×

**Winner: TM** — TM 0.833 (NTEUniform) vs ML 0.808 (ExtraTrees), Δ +0.025


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | NTEUniform | 0.8331 |
| 2 | ResonantGradientBinarizer | 0.8331 |
| 3 | SAQT | 0.8331 |
| 4 | QBEspresso | 0.8078 |
| 5 | SDQB | 0.8078 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| ExtraTrees | 0.8079 |
| RandomForest | 0.7500 |
| LightGBM | 0.6760 |
| XGBoost | 0.6482 |
