# Computers

`time_series` · 2 classes · 350 train / 150 test · 512 features · imbalance 1.0×

**Winner: TM** — TM 0.673 (OnlineQuantileTrackerBinarizer) vs ML 0.633 (RandomForest), Δ +0.040


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | OnlineQuantileTrackerBinarizer | 0.6730 |
| 2 | ResonantGradientBinarizer | 0.6726 |
| 3 | DriftRobustBinarizer | 0.6533 |
| 4 | MovingWindowBinarizerV2 | 0.6533 |
| 5 | SAQT | 0.6533 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| RandomForest | 0.6329 |
| ExtraTrees | 0.6256 |
| LightGBM | 0.6127 |
| XGBoost | 0.5929 |
