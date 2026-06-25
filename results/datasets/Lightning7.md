# Lightning7

`time_series` · 7 classes · 100 train / 43 test · 319 features · imbalance 2.7×

**Winner: TM** — TM 0.731 (SDQB) vs ML 0.667 (RandomForest), Δ +0.064


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | SDQB | 0.7306 |
| 2 | SketchTDigest | 0.7042 |
| 3 | SAQT | 0.6795 |
| 4 | PulseResonanceBinarizer | 0.6763 |
| 5 | MovingWindowBinarizer | 0.6760 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| RandomForest | 0.6671 |
| ExtraTrees | 0.6632 |
| LightGBM | 0.5975 |
| XGBoost | 0.5532 |
