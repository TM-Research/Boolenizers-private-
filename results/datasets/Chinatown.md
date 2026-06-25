# Chinatown

`time_series` · 2 classes · 254 train / 109 test · 24 features · imbalance 2.5×

**Winner: TM** — TM 1.000 (ACFB) vs ML 1.000 (LightGBM), Δ +0.000


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | ACFB | 1.0000 |
| 2 | DecisionTreeBinarizer | 1.0000 |
| 3 | DriftRobustBinarizer | 1.0000 |
| 4 | DualDynamicsBinarizer | 1.0000 |
| 5 | DynamicPulseBinarizer | 1.0000 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| LightGBM | 1.0000 |
| ExtraTrees | 1.0000 |
| RandomForest | 0.9886 |
| XGBoost | 0.9651 |
