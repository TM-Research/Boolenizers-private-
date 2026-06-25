# ElectricDevices

`time_series` · 7 classes · 8001 train / 3000 test · 96 features · imbalance 3.4×

**Winner: ML** — TM 0.737 (StandardBinarizerNative) vs ML 0.769 (XGBoost), Δ -0.033


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | StandardBinarizerNative | 0.7366 |
| 2 | QBEspresso | 0.7338 |
| 3 | SpectralStabilityBinarizer | 0.7328 |
| 4 | SDQB | 0.7321 |
| 5 | NTEBatchQuantile | 0.7317 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| XGBoost | 0.7694 |
| LightGBM | 0.7575 |
| ExtraTrees | 0.7048 |
| RandomForest | 0.6760 |
