# sd-iot

`tabular` · 10 classes · 30000 train / 10000 test · 8 features · imbalance 4.6×

**Winner: ML** — TM 0.995 (KalmanFilterBinarizer) vs ML 0.999 (XGBoost), Δ -0.004


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | KalmanFilterBinarizer | 0.9945 |
| 2 | StandardBinarizerNative | 0.9864 |
| 3 | OnlineQuantileSignalBinarizer | 0.9813 |
| 4 | SDQB | 0.9796 |
| 5 | SAQT | 0.9777 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| XGBoost | 0.9986 |
| RandomForest | 0.9079 |
| ExtraTrees | 0.8831 |
| LightGBM | 0.4002 |
