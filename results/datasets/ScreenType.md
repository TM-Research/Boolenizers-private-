# ScreenType

`time_series` · 3 classes · 525 train / 225 test · 512 features · imbalance 1.0×

**Winner: TM** — TM 0.533 (MovingWindowBinarizerV2) vs ML 0.507 (RandomForest), Δ +0.026


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | MovingWindowBinarizerV2 | 0.5332 |
| 2 | NTEBatchQuantile | 0.5113 |
| 3 | SignalQuantileFusion | 0.5085 |
| 4 | QBEspresso | 0.5078 |
| 5 | KnownMethodsBinarizer | 0.5077 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| RandomForest | 0.5071 |
| ExtraTrees | 0.4839 |
| LightGBM | 0.4425 |
| XGBoost | 0.4355 |
