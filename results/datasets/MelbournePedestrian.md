# MelbournePedestrian

`time_series` · 10 classes · 2543 train / 1090 test · 24 features · imbalance 1.0×

**Winner: TM** — TM 0.910 (GLADEBooleanizer) vs ML 0.900 (ExtraTrees), Δ +0.010


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | GLADEBooleanizer | 0.9096 |
| 2 | OnlineGeneralizedBinarizer | 0.9090 |
| 3 | SpectralStabilityBinarizer | 0.9088 |
| 4 | SAQT | 0.9082 |
| 5 | AdaptiveMomentumBinarizer | 0.9067 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| ExtraTrees | 0.9000 |
| LightGBM | 0.8971 |
| XGBoost | 0.8962 |
| RandomForest | 0.8905 |
