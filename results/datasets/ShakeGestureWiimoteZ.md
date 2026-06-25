# ShakeGestureWiimoteZ

`time_series` · 10 classes · 70 train / 30 test · 385 features · imbalance 1.0×

**Winner: TM** — TM 0.661 (OnlineBollingerBinarizer) vs ML 0.632 (ExtraTrees), Δ +0.029


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | OnlineBollingerBinarizer | 0.6611 |
| 2 | KalmanFilterBinarizer | 0.6311 |
| 3 | DecisionTreeBinarizer | 0.6019 |
| 4 | SSL | 0.6017 |
| 5 | AdaptiveQuantileBinarizer | 0.5957 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| ExtraTrees | 0.6317 |
| RandomForest | 0.5669 |
| XGBoost | 0.5343 |
| LightGBM | 0.4705 |
