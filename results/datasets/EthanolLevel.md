# EthanolLevel

`time_series` · 4 classes · 702 train / 302 test · 512 features · imbalance 1.0×

**Winner: TM** — TM 0.664 (GLADEEncoder) vs ML 0.664 (LightGBM), Δ +0.000


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | GLADEEncoder | 0.6639 |
| 2 | SpectralStabilityBinarizer | 0.6628 |
| 3 | StandardBinarizerNative | 0.6545 |
| 4 | OnlineGeneralizedBinarizer | 0.6477 |
| 5 | GLADEBooleanizer | 0.6432 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| LightGBM | 0.6638 |
| XGBoost | 0.6529 |
| RandomForest | 0.5847 |
| ExtraTrees | 0.5807 |
