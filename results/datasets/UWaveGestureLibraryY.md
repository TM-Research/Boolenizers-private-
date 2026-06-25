# UWaveGestureLibraryY

`time_series` · 8 classes · 3134 train / 1344 test · 315 features · imbalance 1.0×

**Winner: TM** — TM 0.766 (NTEUniform) vs ML 0.760 (RandomForest), Δ +0.006


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | NTEUniform | 0.7662 |
| 2 | StandardBinarizerNative | 0.7631 |
| 3 | OnlineGeneralizedBinarizer | 0.7593 |
| 4 | SAQT | 0.7593 |
| 5 | DecisionTreeBinarizer | 0.7582 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| RandomForest | 0.7605 |
| LightGBM | 0.7598 |
| ExtraTrees | 0.7591 |
| XGBoost | 0.7446 |
