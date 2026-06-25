# FordB

`time_series` · 2 classes · 3112 train / 1334 test · 500 features · imbalance 1.0×

**Winner: ML** — TM 0.788 (SAQT) vs ML 0.788 (XGBoost), Δ -0.000


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | SAQT | 0.7884 |
| 2 | GLADEBooleanizer | 0.7869 |
| 3 | GLADEEncoder | 0.7853 |
| 4 | DriftRobustBinarizer | 0.7844 |
| 5 | ResonantGradientBinarizerV2 | 0.7844 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| XGBoost | 0.7885 |
| LightGBM | 0.7869 |
| RandomForest | 0.7372 |
| ExtraTrees | 0.7323 |
