# DodgerLoopWeekend

`time_series` · 2 classes · 100 train / 44 test · 288 features · imbalance 2.2×

**Winner: TM** — TM 1.000 (ACFB) vs ML 1.000 (XGBoost), Δ +0.000


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | ACFB | 1.0000 |
| 2 | AdaptiveGaussian | 1.0000 |
| 3 | AdaptiveMomentumBinarizer | 1.0000 |
| 4 | AdaptiveQuantileBinarizer | 1.0000 |
| 5 | DecisionTreeBinarizer | 1.0000 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| XGBoost | 1.0000 |
| LightGBM | 1.0000 |
| RandomForest | 1.0000 |
| ExtraTrees | 1.0000 |
