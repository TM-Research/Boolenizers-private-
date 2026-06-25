# InsectEPGSmallTrain

`time_series` · 3 classes · 186 train / 80 test · 512 features · imbalance 2.8×

**Winner: TM** — TM 0.881 (TWINEv3) vs ML 0.840 (ExtraTrees), Δ +0.041


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | TWINEv3 | 0.8806 |
| 2 | SketchGK | 0.8463 |
| 3 | DecisionTreeBinarizer | 0.8300 |
| 4 | SketchTDigest | 0.8300 |
| 5 | ACFB | 0.8247 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| ExtraTrees | 0.8396 |
| RandomForest | 0.8007 |
| XGBoost | 0.7318 |
| LightGBM | 0.7225 |
