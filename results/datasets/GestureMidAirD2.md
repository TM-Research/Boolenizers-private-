# GestureMidAirD2

`time_series` · 26 classes · 236 train / 102 test · 360 features · imbalance 1.1×

**Winner: ML** — TM 0.527 (SAQT) vs ML 0.536 (ExtraTrees), Δ -0.009


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | SAQT | 0.5267 |
| 2 | NTEUniform | 0.5076 |
| 3 | OnlineGeneralizedBinarizer | 0.5045 |
| 4 | SignalQuantileFusion | 0.5010 |
| 5 | GLADEEncoder | 0.5001 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| ExtraTrees | 0.5358 |
| RandomForest | 0.5038 |
| LightGBM | 0.4473 |
| XGBoost | 0.4384 |
