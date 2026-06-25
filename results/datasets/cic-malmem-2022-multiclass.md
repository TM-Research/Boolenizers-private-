# cic-malmem-2022-multiclass

`tabular` · 16 classes · 30001 train / 10000 test · 52 features · imbalance 20.8×

**Winner: ML** — TM 0.362 (GLADEEncoder) vs ML 0.466 (XGBoost), Δ -0.104


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | GLADEEncoder | 0.3622 |
| 2 | SAQT | 0.3619 |
| 3 | GLADEBooleanizer | 0.3576 |
| 4 | KnownMethodsBinarizer | 0.3463 |
| 5 | SignalQuantileFusion | 0.3438 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| XGBoost | 0.4662 |
| LightGBM | 0.4650 |
| RandomForest | 0.4568 |
| ExtraTrees | 0.4568 |
