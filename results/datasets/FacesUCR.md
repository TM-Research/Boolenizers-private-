# FacesUCR

`time_series` · 14 classes · 1575 train / 675 test · 131 features · imbalance 6.7×

**Winner: TM** — TM 0.962 (KBinsThermometer) vs ML 0.943 (ExtraTrees), Δ +0.019


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | KBinsThermometer | 0.9620 |
| 2 | KalmanFilterBinarizer | 0.9614 |
| 3 | ResonantGradientBinarizerV2 | 0.9612 |
| 4 | SingleSpeedP2 | 0.9612 |
| 5 | SketchTDigest | 0.9597 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| ExtraTrees | 0.9428 |
| LightGBM | 0.9403 |
| XGBoost | 0.9290 |
| RandomForest | 0.9179 |
