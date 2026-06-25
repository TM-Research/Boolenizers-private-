# Meat

`time_series` · 3 classes · 84 train / 36 test · 448 features · imbalance 1.0×

**Winner: TM** — TM 1.000 (ACFB) vs ML 1.000 (LightGBM), Δ +0.000


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | ACFB | 1.0000 |
| 2 | AdaptiveQuantileBinarizer | 1.0000 |
| 3 | DecisionTreeBinarizer | 1.0000 |
| 4 | DualDynamicsBinarizer | 1.0000 |
| 5 | DynamicPulseBinarizer | 1.0000 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| LightGBM | 1.0000 |
| RandomForest | 1.0000 |
| XGBoost | 0.9722 |
| ExtraTrees | 0.9722 |
