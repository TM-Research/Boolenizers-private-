# PigAirwayPressure

`time_series` · 52 classes · 218 train / 94 test · 512 features · imbalance 1.2×

**Winner: TM** — TM 0.103 (ACFB) vs ML 0.097 (ExtraTrees), Δ +0.006


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | ACFB | 0.1026 |
| 2 | DecisionTreeBinarizer | 0.0757 |
| 3 | SpectralStabilityBinarizer | 0.0472 |
| 4 | DynamicPulseBinarizer | 0.0462 |
| 5 | SSL | 0.0406 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| ExtraTrees | 0.0966 |
| RandomForest | 0.0701 |
| XGBoost | 0.0542 |
| LightGBM | 0.0452 |
