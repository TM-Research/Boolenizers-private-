# BeetleFly

`time_series` · 2 classes · 28 train / 12 test · 512 features · imbalance 1.0×

**Winner: TM** — TM 0.916 (MWAB) vs ML 0.916 (XGBoost), Δ +0.000


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | MWAB | 0.9161 |
| 2 | ResonantGradientBinarizerV2 | 0.9161 |
| 3 | TWINEv3 | 0.9161 |
| 4 | AdaptiveGaussian | 0.8333 |
| 5 | OnlineATRBinarizer | 0.8333 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| XGBoost | 0.9161 |
| ExtraTrees | 0.9161 |
| RandomForest | 0.7483 |
| LightGBM | 0.3333 |
