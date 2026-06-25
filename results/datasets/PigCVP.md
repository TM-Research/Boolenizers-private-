# PigCVP

`time_series` · 52 classes · 218 train / 94 test · 512 features · imbalance 1.2×

**Winner: ML** — TM 0.017 (SingleSpeedP2) vs ML 0.125 (ExtraTrees), Δ -0.109


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | SingleSpeedP2 | 0.0166 |
| 2 | SpectralStabilityBinarizer | 0.0147 |
| 3 | SSL | 0.0135 |
| 4 | OnlineQuantileTrackerBinarizer | 0.0085 |
| 5 | NTEBatchQuantile | 0.0064 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| ExtraTrees | 0.1254 |
| RandomForest | 0.0308 |
| LightGBM | 0.0299 |
| XGBoost | 0.0096 |
