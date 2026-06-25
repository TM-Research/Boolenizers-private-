# HandOutlines

`time_series` · 2 classes · 959 train / 411 test · 512 features · imbalance 1.8×

**Winner: TM** — TM 0.920 (DriftRobustBinarizer) vs ML 0.918 (RandomForest), Δ +0.002


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | DriftRobustBinarizer | 0.9203 |
| 2 | ResonantGradientBinarizer | 0.9198 |
| 3 | SketchTDigest | 0.9180 |
| 4 | SketchGK | 0.9142 |
| 5 | SingleSpeedP2 | 0.9128 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| RandomForest | 0.9180 |
| XGBoost | 0.9122 |
| ExtraTrees | 0.9122 |
| LightGBM | 0.9013 |
