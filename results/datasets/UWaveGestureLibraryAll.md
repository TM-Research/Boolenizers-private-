# UWaveGestureLibraryAll

`time_series` · 8 classes · 3134 train / 1344 test · 512 features · imbalance 1.0×

**Winner: TM** — TM 0.978 (DecisionTreeBinarizer) vs ML 0.972 (ExtraTrees), Δ +0.006


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | DecisionTreeBinarizer | 0.9776 |
| 2 | MovingWindowBinarizerV2 | 0.9761 |
| 3 | KnownMethodsBinarizer | 0.9754 |
| 4 | QBEspresso | 0.9753 |
| 5 | NTEUniform | 0.9747 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| ExtraTrees | 0.9716 |
| LightGBM | 0.9612 |
| RandomForest | 0.9605 |
| XGBoost | 0.9561 |
