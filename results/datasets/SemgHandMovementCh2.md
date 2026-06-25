# SemgHandMovementCh2

`time_series` · 6 classes · 630 train / 270 test · 512 features · imbalance 1.0×

**Winner: TM** — TM 0.521 (PulseResonanceBinarizer) vs ML 0.518 (ExtraTrees), Δ +0.003


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | PulseResonanceBinarizer | 0.5210 |
| 2 | DecisionTreeBinarizer | 0.5203 |
| 3 | AdaptiveQuantileBinarizer | 0.5124 |
| 4 | KBinsThermometer | 0.5108 |
| 5 | KnownMethodsBinarizer | 0.5103 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| ExtraTrees | 0.5180 |
| XGBoost | 0.4833 |
| LightGBM | 0.4620 |
| RandomForest | 0.4599 |
