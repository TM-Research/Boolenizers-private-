# SyntheticControl

`time_series` · 6 classes · 420 train / 180 test · 60 features · imbalance 1.0×

**Winner: ML** — TM 0.978 (GLADEEncoder) vs ML 0.994 (RandomForest), Δ -0.016


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | GLADEEncoder | 0.9780 |
| 2 | DualDynamicsBinarizer | 0.9779 |
| 3 | KBinsThermometer | 0.9779 |
| 4 | GLADEBooleanizer | 0.9724 |
| 5 | KnownMethodsBinarizer | 0.9723 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| RandomForest | 0.9944 |
| ExtraTrees | 0.9833 |
| LightGBM | 0.9268 |
| XGBoost | 0.9033 |
