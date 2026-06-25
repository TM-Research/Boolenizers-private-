# ToeSegmentation1

`time_series` · 2 classes · 187 train / 81 test · 277 features · imbalance 1.1×

**Winner: TM** — TM 0.839 (OGBFast) vs ML 0.797 (ExtraTrees), Δ +0.042


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | OGBFast | 0.8391 |
| 2 | OnlineGeneralizedBinarizer | 0.8391 |
| 3 | KBinsThermometer | 0.8386 |
| 4 | SpectralStabilityBinarizer | 0.8271 |
| 5 | MovingWindowBinarizerV2 | 0.8269 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| ExtraTrees | 0.7972 |
| RandomForest | 0.7750 |
| XGBoost | 0.7274 |
| LightGBM | 0.7264 |
