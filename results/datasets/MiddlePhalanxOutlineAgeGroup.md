# MiddlePhalanxOutlineAgeGroup

`time_series` · 3 classes · 387 train / 167 test · 80 features · imbalance 2.9×

**Winner: ML** — TM 0.699 (OnlineUniversalBinarizer) vs ML 0.704 (ExtraTrees), Δ -0.005


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | OnlineUniversalBinarizer | 0.6993 |
| 2 | OGBFast | 0.6978 |
| 3 | OnlineQuantileSignalBinarizer | 0.6959 |
| 4 | OnlineDeltaMomentumBinarizer | 0.6834 |
| 5 | OnlineRSIMACDBinarizer | 0.6825 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| ExtraTrees | 0.7039 |
| RandomForest | 0.6993 |
| XGBoost | 0.6961 |
| LightGBM | 0.6740 |
