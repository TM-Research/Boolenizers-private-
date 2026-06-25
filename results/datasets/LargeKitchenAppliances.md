# LargeKitchenAppliances

`time_series` · 3 classes · 525 train / 225 test · 512 features · imbalance 1.0×

**Winner: TM** — TM 0.666 (ResonantGradientBinarizerV2) vs ML 0.641 (LightGBM), Δ +0.024


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | ResonantGradientBinarizerV2 | 0.6656 |
| 2 | OGBFast | 0.6523 |
| 3 | NTEBatchQuantile | 0.6459 |
| 4 | SpectralStabilityBinarizer | 0.6433 |
| 5 | MovingWindowBinarizer | 0.6430 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| LightGBM | 0.6412 |
| ExtraTrees | 0.6252 |
| XGBoost | 0.6219 |
| RandomForest | 0.6154 |
