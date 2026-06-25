# RefrigerationDevices

`time_series` · 3 classes · 525 train / 225 test · 512 features · imbalance 1.0×

**Winner: TM** — TM 0.600 (OnlineGeneralizedBinarizer) vs ML 0.598 (RandomForest), Δ +0.002


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | OnlineGeneralizedBinarizer | 0.6000 |
| 2 | GLADEBooleanizer | 0.5990 |
| 3 | StandardBinarizerNative | 0.5983 |
| 4 | TWINELite | 0.5973 |
| 5 | SpectralStabilityBinarizer | 0.5954 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| RandomForest | 0.5977 |
| ExtraTrees | 0.5954 |
| LightGBM | 0.5767 |
| XGBoost | 0.5711 |
