# Worms

`time_series` · 5 classes · 180 train / 78 test · 512 features · imbalance 4.2×

**Winner: TM** — TM 0.552 (SDQB) vs ML 0.505 (ExtraTrees), Δ +0.047


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | SDQB | 0.5518 |
| 2 | PulseResonanceBinarizer | 0.5449 |
| 3 | KnownMethodsBinarizer | 0.5284 |
| 4 | DecisionTreeBinarizer | 0.5250 |
| 5 | SketchTDigest | 0.5248 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| ExtraTrees | 0.5051 |
| RandomForest | 0.4750 |
| LightGBM | 0.4738 |
| XGBoost | 0.4308 |
