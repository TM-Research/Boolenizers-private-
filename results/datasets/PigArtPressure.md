# PigArtPressure

`time_series` · 52 classes · 218 train / 94 test · 512 features · imbalance 1.2×

**Winner: ML** — TM 0.075 (SingleSpeedP2) vs ML 0.143 (ExtraTrees), Δ -0.069


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | SingleSpeedP2 | 0.0747 |
| 2 | PulseResonanceBinarizer | 0.0636 |
| 3 | StandardBinarizerNative | 0.0519 |
| 4 | SignalGradientBinarizer | 0.0513 |
| 5 | SDQB | 0.0497 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| ExtraTrees | 0.1433 |
| RandomForest | 0.0638 |
| XGBoost | 0.0494 |
| LightGBM | 0.0282 |
