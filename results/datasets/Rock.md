# Rock

`time_series` · 4 classes · 49 train / 21 test · 512 features · imbalance 2.2×

**Winner: TM** — TM 0.922 (ACFB) vs ML 0.922 (LightGBM), Δ +0.000


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | ACFB | 0.9222 |
| 2 | AdaptiveMomentumBinarizer | 0.9222 |
| 3 | AdaptiveQuantileBinarizer | 0.9222 |
| 4 | DecisionTreeBinarizer | 0.9222 |
| 5 | DualDynamicsBinarizer | 0.9222 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| LightGBM | 0.9222 |
| ExtraTrees | 0.9222 |
| RandomForest | 0.8605 |
| XGBoost | 0.8508 |
