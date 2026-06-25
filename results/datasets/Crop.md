# Crop

`time_series` · 24 classes · 7992 train / 3000 test · 46 features · imbalance 1.0×

**Winner: TM** — TM 0.756 (QBEspresso) vs ML 0.742 (XGBoost), Δ +0.014


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | QBEspresso | 0.7560 |
| 2 | StandardBinarizerNative | 0.7556 |
| 3 | SAQT | 0.7555 |
| 4 | AdaptiveMomentumBinarizer | 0.7515 |
| 5 | KBinsThermometer | 0.7511 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| XGBoost | 0.7420 |
| LightGBM | 0.7357 |
| ExtraTrees | 0.7339 |
| RandomForest | 0.7219 |
