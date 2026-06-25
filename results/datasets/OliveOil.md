# OliveOil

`time_series` · 4 classes · 42 train / 18 test · 512 features · imbalance 2.8×

**Winner: TM** — TM 1.000 (SignalQuantileFusion) vs ML 1.000 (LightGBM), Δ +0.000


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | SignalQuantileFusion | 1.0000 |
| 2 | AdaptiveMomentumBinarizer | 0.9020 |
| 3 | AdaptiveQuantileBinarizer | 0.9020 |
| 4 | DecisionTreeBinarizer | 0.9020 |
| 5 | DynamicPulseBinarizer | 0.9020 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| LightGBM | 1.0000 |
| RandomForest | 1.0000 |
| ExtraTrees | 1.0000 |
| XGBoost | 0.7994 |
