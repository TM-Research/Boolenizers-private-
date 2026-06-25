# Wafer

`time_series` · 2 classes · 5014 train / 2150 test · 152 features · imbalance 8.4×

**Winner: TM** — TM 0.999 (AdaptiveQuantileBinarizer) vs ML 0.998 (XGBoost), Δ +0.001


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | AdaptiveQuantileBinarizer | 0.9988 |
| 2 | GLADEBooleanizer | 0.9988 |
| 3 | GLADEEncoder | 0.9988 |
| 4 | KalmanFilterBinarizer | 0.9988 |
| 5 | KnownMethodsBinarizer | 0.9988 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| XGBoost | 0.9976 |
| LightGBM | 0.9976 |
| ExtraTrees | 0.9976 |
| RandomForest | 0.9915 |
