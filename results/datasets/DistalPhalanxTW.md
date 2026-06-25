# DistalPhalanxTW

`time_series` · 6 classes · 377 train / 162 test · 80 features · imbalance 9.9×

**Winner: ML** — TM 0.508 (DriftRobustBinarizer) vs ML 0.550 (XGBoost), Δ -0.042


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | DriftRobustBinarizer | 0.5080 |
| 2 | PulseResonanceBinarizer | 0.5045 |
| 3 | GLADEEncoder | 0.4999 |
| 4 | SignalGradientBinarizer | 0.4953 |
| 5 | MovingWindowBinarizerV2 | 0.4936 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| XGBoost | 0.5502 |
| ExtraTrees | 0.4933 |
| RandomForest | 0.4850 |
| LightGBM | 0.4805 |
