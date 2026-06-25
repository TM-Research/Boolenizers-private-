# DiatomSizeReduction

`time_series` · 4 classes · 225 train / 97 test · 345 features · imbalance 2.9×

**Winner: TM** — TM 1.000 (ACFB) vs ML 1.000 (RandomForest), Δ +0.000


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | ACFB | 1.0000 |
| 2 | AdaptiveQuantileBinarizer | 1.0000 |
| 3 | DynamicPulseBinarizer | 1.0000 |
| 4 | MovingWindowBinarizer | 1.0000 |
| 5 | ResonantGradientBinarizer | 1.0000 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| RandomForest | 1.0000 |
| ExtraTrees | 1.0000 |
| XGBoost | 0.9642 |
| LightGBM | 0.9642 |
