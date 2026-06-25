# Phoneme

`time_series` · 39 classes · 1477 train / 633 test · 512 features · imbalance 167.0×

**Winner: TM** — TM 0.099 (DualDynamicsBinarizer) vs ML 0.064 (XGBoost), Δ +0.035


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | DualDynamicsBinarizer | 0.0989 |
| 2 | StandardBinarizerWrapper | 0.0899 |
| 3 | TWINEv3 | 0.0869 |
| 4 | AdaptiveGaussian | 0.0847 |
| 5 | DecisionTreeBinarizer | 0.0805 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| XGBoost | 0.0639 |
| ExtraTrees | 0.0527 |
| RandomForest | 0.0488 |
| LightGBM | 0.0081 |
