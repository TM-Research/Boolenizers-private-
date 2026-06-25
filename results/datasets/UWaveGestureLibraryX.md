# UWaveGestureLibraryX

`time_series` · 8 classes · 3134 train / 1344 test · 315 features · imbalance 1.0×

**Winner: TM** — TM 0.831 (SketchTDigest) vs ML 0.815 (ExtraTrees), Δ +0.016


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | SketchTDigest | 0.8313 |
| 2 | ACFB | 0.8274 |
| 3 | SignalQuantileFusion | 0.8238 |
| 4 | OnlineGeneralizedBinarizer | 0.8233 |
| 5 | PulseResonanceBinarizer | 0.8232 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| ExtraTrees | 0.8150 |
| XGBoost | 0.8031 |
| LightGBM | 0.8017 |
| RandomForest | 0.8012 |
