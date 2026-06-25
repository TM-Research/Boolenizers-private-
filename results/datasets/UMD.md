# UMD

`time_series` · 3 classes · 126 train / 54 test · 150 features · imbalance 1.0×

**Winner: TM** — TM 1.000 (ACFB) vs ML 1.000 (ExtraTrees), Δ +0.000


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | ACFB | 1.0000 |
| 2 | AdaptiveMomentumBinarizer | 1.0000 |
| 3 | DynamicPulseBinarizer | 1.0000 |
| 4 | NTEUniform | 1.0000 |
| 5 | OGBFast | 1.0000 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| ExtraTrees | 1.0000 |
| LightGBM | 0.9815 |
| RandomForest | 0.9815 |
| XGBoost | 0.9629 |
