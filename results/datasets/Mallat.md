# Mallat

`time_series` · 8 classes · 1680 train / 720 test · 512 features · imbalance 1.0×

**Winner: TM** — TM 0.999 (DualDynamicsBinarizer) vs ML 0.996 (RandomForest), Δ +0.003


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | DualDynamicsBinarizer | 0.9986 |
| 2 | SSL | 0.9986 |
| 3 | OnlineBollingerBinarizer | 0.9972 |
| 4 | ACFB | 0.9958 |
| 5 | OnlineATRBinarizer | 0.9958 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| RandomForest | 0.9958 |
| ExtraTrees | 0.9944 |
| LightGBM | 0.9917 |
| XGBoost | 0.9903 |
