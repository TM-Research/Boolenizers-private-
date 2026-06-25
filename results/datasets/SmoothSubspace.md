# SmoothSubspace

`time_series` · 3 classes · 210 train / 90 test · 15 features · imbalance 1.0×

**Winner: TM** — TM 0.989 (AdaptiveMomentumBinarizer) vs ML 0.989 (ExtraTrees), Δ +0.000


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | AdaptiveMomentumBinarizer | 0.9889 |
| 2 | DriftRobustBinarizer | 0.9889 |
| 3 | DualDynamicsBinarizer | 0.9889 |
| 4 | DynamicPulseBinarizer | 0.9889 |
| 5 | GLADEBooleanizer | 0.9889 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| ExtraTrees | 0.9889 |
| LightGBM | 0.9668 |
| RandomForest | 0.9559 |
| XGBoost | 0.9336 |
