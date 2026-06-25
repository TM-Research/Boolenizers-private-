# safe-advent-2025

`tabular` · 4 classes · 30000 train / 10000 test · 17 features · imbalance 13428.0×

**Winner: TM** — TM 0.495 (PulseResonanceBinarizer) vs ML 0.355 (ExtraTrees), Δ +0.140


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | PulseResonanceBinarizer | 0.4951 |
| 2 | AQB | 0.4754 |
| 3 | SketchTDigest | 0.4377 |
| 4 | GLADEEncoder | 0.4259 |
| 5 | SingleSpeedP2 | 0.3968 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| ExtraTrees | 0.3547 |
| RandomForest | 0.3539 |
| LightGBM | 0.3535 |
| XGBoost | 0.2083 |
