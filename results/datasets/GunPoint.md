# GunPoint

`time_series` · 2 classes · 140 train / 60 test · 150 features · imbalance 1.0×

**Winner: TM** — TM 0.967 (AdaptiveQuantileBinarizer) vs ML 0.967 (RandomForest), Δ +0.000


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | AdaptiveQuantileBinarizer | 0.9667 |
| 2 | DriftRobustBinarizer | 0.9667 |
| 3 | DualDynamicsBinarizer | 0.9667 |
| 4 | KBinsThermometer | 0.9667 |
| 5 | KnownMethodsBinarizer | 0.9667 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| RandomForest | 0.9667 |
| ExtraTrees | 0.9667 |
| LightGBM | 0.9500 |
| XGBoost | 0.9333 |
