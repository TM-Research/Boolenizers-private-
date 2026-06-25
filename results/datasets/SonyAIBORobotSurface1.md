# SonyAIBORobotSurface1

`time_series` · 2 classes · 434 train / 187 test · 70 features · imbalance 1.3×

**Winner: TM** — TM 1.000 (GLADEBooleanizer) vs ML 0.995 (ExtraTrees), Δ +0.005


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | GLADEBooleanizer | 1.0000 |
| 2 | NTEBatchQuantile | 1.0000 |
| 3 | StandardBinarizerWrapper | 1.0000 |
| 4 | AdaptiveMomentumBinarizer | 0.9946 |
| 5 | AdaptiveQuantileBinarizer | 0.9946 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| ExtraTrees | 0.9946 |
| LightGBM | 0.9892 |
| RandomForest | 0.9836 |
| XGBoost | 0.9619 |
