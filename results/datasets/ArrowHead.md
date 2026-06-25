# ArrowHead

`time_series` · 3 classes · 147 train / 64 test · 251 features · imbalance 1.3×

**Winner: TM** — TM 0.939 (ACFB) vs ML 0.908 (RandomForest), Δ +0.031


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | ACFB | 0.9385 |
| 2 | ResonantGradientBinarizerV2 | 0.9385 |
| 3 | DynamicPulseBinarizer | 0.9241 |
| 4 | KalmanFilterBinarizer | 0.9241 |
| 5 | SignalQuantileFusion | 0.9241 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| RandomForest | 0.9078 |
| ExtraTrees | 0.8903 |
| LightGBM | 0.8764 |
| XGBoost | 0.8127 |
