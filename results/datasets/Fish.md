# Fish

`time_series` · 7 classes · 245 train / 105 test · 463 features · imbalance 1.0×

**Winner: TM** — TM 0.971 (MovingWindowBinarizer) vs ML 0.953 (ExtraTrees), Δ +0.019


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | MovingWindowBinarizer | 0.9714 |
| 2 | KnownMethodsBinarizer | 0.9622 |
| 3 | SignalGradientBinarizer | 0.9619 |
| 4 | SignalQuantileFusion | 0.9618 |
| 5 | ACFB | 0.9524 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| ExtraTrees | 0.9526 |
| RandomForest | 0.9232 |
| LightGBM | 0.8649 |
| XGBoost | 0.8542 |
