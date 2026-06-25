# InsectWingbeatSound

`time_series` · 11 classes · 1540 train / 660 test · 256 features · imbalance 1.0×

**Winner: TM** — TM 0.708 (GLADEEncoder) vs ML 0.684 (LightGBM), Δ +0.023


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | GLADEEncoder | 0.7075 |
| 2 | OnlineQuantileTrackerBinarizer | 0.7075 |
| 3 | StandardBinarizerNative | 0.7066 |
| 4 | DriftRobustBinarizer | 0.7050 |
| 5 | OnlineGeneralizedBinarizer | 0.7036 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| LightGBM | 0.6843 |
| XGBoost | 0.6821 |
| ExtraTrees | 0.6743 |
| RandomForest | 0.6553 |
