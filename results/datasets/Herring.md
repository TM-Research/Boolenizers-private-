# Herring

`time_series` · 2 classes · 89 train / 39 test · 512 features · imbalance 1.5×

**Winner: ML** — TM 0.654 (DynamicPulseBinarizer) vs ML 0.666 (XGBoost), Δ -0.012


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | DynamicPulseBinarizer | 0.6538 |
| 2 | GLADEBooleanizer | 0.6538 |
| 3 | AdaptiveMomentumBinarizer | 0.6432 |
| 4 | DualDynamicsBinarizer | 0.6432 |
| 5 | KnownMethodsBinarizer | 0.6389 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| XGBoost | 0.6657 |
| LightGBM | 0.6100 |
| RandomForest | 0.5585 |
| ExtraTrees | 0.4759 |
