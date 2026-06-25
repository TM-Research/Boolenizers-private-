# FaceAll

`time_series` · 14 classes · 1575 train / 675 test · 131 features · imbalance 6.7×

**Winner: TM** — TM 0.969 (DecisionTreeBinarizer) vs ML 0.946 (ExtraTrees), Δ +0.023


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | DecisionTreeBinarizer | 0.9693 |
| 2 | TWINEv3 | 0.9692 |
| 3 | TWINELite | 0.9686 |
| 4 | OnlineQuantileSignalBinarizer | 0.9680 |
| 5 | ResonantGradientBinarizerV2 | 0.9678 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| ExtraTrees | 0.9460 |
| RandomForest | 0.9326 |
| LightGBM | 0.9302 |
| XGBoost | 0.9222 |
