# Fungi

`time_series` · 18 classes · 142 train / 62 test · 201 features · imbalance 2.8×

**Winner: TM** — TM 1.000 (AdaptiveMomentumBinarizer) vs ML 1.000 (RandomForest), Δ +0.000


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | AdaptiveMomentumBinarizer | 1.0000 |
| 2 | DecisionTreeBinarizer | 1.0000 |
| 3 | GLADEEncoder | 1.0000 |
| 4 | KalmanFilterBinarizer | 1.0000 |
| 5 | OnlineQuantileTrackerBinarizer | 1.0000 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| RandomForest | 1.0000 |
| ExtraTrees | 1.0000 |
| LightGBM | 0.9013 |
| XGBoost | 0.7947 |
