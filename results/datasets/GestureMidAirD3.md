# GestureMidAirD3

`time_series` · 26 classes · 236 train / 102 test · 360 features · imbalance 1.1×

**Winner: TM** — TM 0.422 (SketchGK) vs ML 0.346 (ExtraTrees), Δ +0.076


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | SketchGK | 0.4219 |
| 2 | SDQB | 0.4204 |
| 3 | DynamicPulseBinarizer | 0.4086 |
| 4 | SingleSpeedP2 | 0.4046 |
| 5 | DriftRobustBinarizer | 0.4044 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| ExtraTrees | 0.3457 |
| RandomForest | 0.3371 |
| XGBoost | 0.3271 |
| LightGBM | 0.3093 |
