# MiddlePhalanxTW

`time_series` · 6 classes · 387 train / 166 test · 80 features · imbalance 5.9×

**Winner: TM** — TM 0.422 (OnlineBollingerBinarizer) vs ML 0.383 (RandomForest), Δ +0.039


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | OnlineBollingerBinarizer | 0.4222 |
| 2 | ACFB | 0.4099 |
| 3 | DecisionTreeBinarizer | 0.4038 |
| 4 | MovingWindowBinarizer | 0.3853 |
| 5 | OnlineGeneralizedBinarizer | 0.3825 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| RandomForest | 0.3831 |
| ExtraTrees | 0.3744 |
| XGBoost | 0.3673 |
| LightGBM | 0.3460 |
