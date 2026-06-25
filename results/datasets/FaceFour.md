# FaceFour

`time_series` · 4 classes · 78 train / 34 test · 350 features · imbalance 1.6×

**Winner: TM** — TM 1.000 (AdaptiveQuantileBinarizer) vs ML 0.971 (XGBoost), Δ +0.029


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | AdaptiveQuantileBinarizer | 1.0000 |
| 2 | MovingWindowBinarizerV2 | 1.0000 |
| 3 | DriftRobustBinarizer | 0.9737 |
| 4 | ACFB | 0.9714 |
| 5 | AdaptiveMomentumBinarizer | 0.9686 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| XGBoost | 0.9706 |
| LightGBM | 0.9686 |
| RandomForest | 0.9686 |
| ExtraTrees | 0.9686 |
