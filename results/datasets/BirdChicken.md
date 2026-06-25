# BirdChicken

`time_series` · 2 classes · 28 train / 12 test · 512 features · imbalance 1.0×

**Winner: ML** — TM 0.916 (SDQB) vs ML 1.000 (XGBoost), Δ -0.084


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | SDQB | 0.9161 |
| 2 | ResonantGradientBinarizer | 0.8333 |
| 3 | SketchTDigest | 0.8333 |
| 4 | AdaptiveQuantileBinarizer | 0.7483 |
| 5 | KBinsThermometer | 0.7483 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| XGBoost | 1.0000 |
| RandomForest | 0.7333 |
| ExtraTrees | 0.6571 |
| LightGBM | 0.3333 |
