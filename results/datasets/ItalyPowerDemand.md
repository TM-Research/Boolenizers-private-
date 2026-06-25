# ItalyPowerDemand

`time_series` · 2 classes · 767 train / 329 test · 24 features · imbalance 1.0×

**Winner: TM** — TM 0.973 (OnlineATRBinarizer) vs ML 0.970 (RandomForest), Δ +0.003


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | OnlineATRBinarizer | 0.9726 |
| 2 | DecisionTreeBinarizer | 0.9696 |
| 3 | MovingWindowBinarizerV2 | 0.9696 |
| 4 | ResonantGradientBinarizer | 0.9696 |
| 5 | ResonantGradientBinarizerV2 | 0.9696 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| RandomForest | 0.9696 |
| ExtraTrees | 0.9696 |
| LightGBM | 0.9635 |
| XGBoost | 0.9514 |
