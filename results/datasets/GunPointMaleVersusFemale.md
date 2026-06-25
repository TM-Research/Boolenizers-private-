# GunPointMaleVersusFemale

`time_series` · 2 classes · 315 train / 136 test · 150 features · imbalance 1.1×

**Winner: TM** — TM 1.000 (ACFB) vs ML 0.993 (ExtraTrees), Δ +0.007


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | ACFB | 1.0000 |
| 2 | AdaptiveMomentumBinarizer | 0.9926 |
| 3 | DecisionTreeBinarizer | 0.9926 |
| 4 | GLADEBooleanizer | 0.9926 |
| 5 | KBinsThermometer | 0.9926 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| ExtraTrees | 0.9926 |
| XGBoost | 0.9853 |
| RandomForest | 0.9779 |
| LightGBM | 0.9778 |
