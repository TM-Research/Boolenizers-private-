# ACSF1

`time_series` · 10 classes · 140 train / 60 test · 512 features · imbalance 1.0×

**Winner: TM** — TM 0.831 (GLADEEncoder) vs ML 0.803 (ExtraTrees), Δ +0.027


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | GLADEEncoder | 0.8308 |
| 2 | MovingWindowBinarizerV2 | 0.8285 |
| 3 | DecisionTreeBinarizer | 0.8129 |
| 4 | NTEUniform | 0.8096 |
| 5 | SpectralStabilityBinarizer | 0.7982 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| ExtraTrees | 0.8035 |
| RandomForest | 0.7821 |
| XGBoost | 0.7781 |
| LightGBM | 0.6673 |
