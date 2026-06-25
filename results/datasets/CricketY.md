# CricketY

`time_series` · 12 classes · 546 train / 234 test · 300 features · imbalance 1.0×

**Winner: TM** — TM 0.701 (OnlineQuantileTrackerBinarizer) vs ML 0.684 (ExtraTrees), Δ +0.017


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | OnlineQuantileTrackerBinarizer | 0.7007 |
| 2 | SpectralStabilityBinarizer | 0.6985 |
| 3 | TWINELite | 0.6936 |
| 4 | SketchGK | 0.6894 |
| 5 | StandardBinarizerWrapper | 0.6869 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| ExtraTrees | 0.6838 |
| RandomForest | 0.6684 |
| LightGBM | 0.6359 |
| XGBoost | 0.6187 |
