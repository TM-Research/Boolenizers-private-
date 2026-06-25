# CinCECGTorso

`time_series` · 4 classes · 994 train / 426 test · 512 features · imbalance 1.0×

**Winner: TM** — TM 1.000 (AdaptiveQuantileBinarizer) vs ML 0.998 (LightGBM), Δ +0.002


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | AdaptiveQuantileBinarizer | 1.0000 |
| 2 | DecisionTreeBinarizer | 1.0000 |
| 3 | GLADEEncoder | 1.0000 |
| 4 | KalmanFilterBinarizer | 1.0000 |
| 5 | MWAB | 1.0000 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| LightGBM | 0.9977 |
| ExtraTrees | 0.9977 |
| RandomForest | 0.9953 |
| XGBoost | 0.9859 |
