# Beef

`time_series` · 5 classes · 42 train / 18 test · 470 features · imbalance 1.1×

**Winner: TM** — TM 0.743 (ResonantGradientBinarizerV2) vs ML 0.640 (LightGBM), Δ +0.102


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | ResonantGradientBinarizerV2 | 0.7425 |
| 2 | ACFB | 0.7148 |
| 3 | NTEBatchQuantile | 0.7002 |
| 4 | StandardBinarizerNative | 0.7002 |
| 5 | KBinsThermometer | 0.6524 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| LightGBM | 0.6400 |
| RandomForest | 0.6235 |
| XGBoost | 0.5814 |
| ExtraTrees | 0.5759 |
