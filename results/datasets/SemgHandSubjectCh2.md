# SemgHandSubjectCh2

`time_series` · 5 classes · 630 train / 270 test · 512 features · imbalance 1.0×

**Winner: TM** — TM 0.797 (OnlineBollingerBinarizer) vs ML 0.707 (LightGBM), Δ +0.091


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | OnlineBollingerBinarizer | 0.7973 |
| 2 | OGBFast | 0.7940 |
| 3 | SAQT | 0.7880 |
| 4 | OnlineQuantileTrackerBinarizer | 0.7861 |
| 5 | KBinsThermometer | 0.7834 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| LightGBM | 0.7067 |
| RandomForest | 0.6951 |
| ExtraTrees | 0.6917 |
| XGBoost | 0.6606 |
