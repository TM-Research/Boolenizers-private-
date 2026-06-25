# Strawberry

`time_series` · 2 classes · 688 train / 295 test · 235 features · imbalance 1.8×

**Winner: TM** — TM 0.978 (OGBFast) vs ML 0.978 (LightGBM), Δ +0.000


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | OGBFast | 0.9780 |
| 2 | StandardBinarizerWrapper | 0.9780 |
| 3 | AdaptiveQuantileBinarizer | 0.9744 |
| 4 | DriftRobustBinarizer | 0.9744 |
| 5 | DualDynamicsBinarizer | 0.9744 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| LightGBM | 0.9780 |
| RandomForest | 0.9743 |
| XGBoost | 0.9707 |
| ExtraTrees | 0.9671 |
