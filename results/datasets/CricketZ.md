# CricketZ

`time_series` · 12 classes · 546 train / 234 test · 300 features · imbalance 1.0×

**Winner: TM** — TM 0.707 (AdaptiveQuantileBinarizer) vs ML 0.684 (ExtraTrees), Δ +0.022


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | AdaptiveQuantileBinarizer | 0.7065 |
| 2 | ResonantGradientBinarizer | 0.7051 |
| 3 | TWINELite | 0.6956 |
| 4 | SDQB | 0.6816 |
| 5 | DriftRobustBinarizer | 0.6810 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| ExtraTrees | 0.6842 |
| RandomForest | 0.6678 |
| LightGBM | 0.6569 |
| XGBoost | 0.5950 |
