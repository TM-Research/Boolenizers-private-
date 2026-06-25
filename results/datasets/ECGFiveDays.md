# ECGFiveDays

`time_series` · 2 classes · 618 train / 266 test · 136 features · imbalance 1.0×

**Winner: TM** — TM 1.000 (AdaptiveQuantileBinarizer) vs ML 0.996 (LightGBM), Δ +0.004


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | AdaptiveQuantileBinarizer | 1.0000 |
| 2 | DriftRobustBinarizer | 1.0000 |
| 3 | GLADEEncoder | 1.0000 |
| 4 | KnownMethodsBinarizer | 1.0000 |
| 5 | OGBFast | 1.0000 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| LightGBM | 0.9962 |
| ExtraTrees | 0.9962 |
| XGBoost | 0.9812 |
| RandomForest | 0.9737 |
