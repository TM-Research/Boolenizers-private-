# MoteStrain

`time_series` · 2 classes · 890 train / 382 test · 84 features · imbalance 1.2×

**Winner: ML** — TM 0.968 (DriftRobustBinarizer) vs ML 0.971 (ExtraTrees), Δ -0.003


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | DriftRobustBinarizer | 0.9684 |
| 2 | SAQT | 0.9684 |
| 3 | SignalGradientBinarizer | 0.9684 |
| 4 | GLADEEncoder | 0.9632 |
| 5 | SDQB | 0.9632 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| ExtraTrees | 0.9710 |
| RandomForest | 0.9579 |
| LightGBM | 0.9552 |
| XGBoost | 0.9421 |
