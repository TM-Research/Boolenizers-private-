# PhalangesOutlinesCorrect

`time_series` · 2 classes · 1860 train / 798 test · 80 features · imbalance 1.8×

**Winner: TM** — TM 0.822 (GLADEEncoder) vs ML 0.819 (RandomForest), Δ +0.003


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | GLADEEncoder | 0.8215 |
| 2 | StandardBinarizerNative | 0.8209 |
| 3 | QBEspresso | 0.8149 |
| 4 | MovingWindowBinarizerV2 | 0.8093 |
| 5 | SignalQuantileFusion | 0.8002 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| RandomForest | 0.8189 |
| LightGBM | 0.8150 |
| ExtraTrees | 0.8078 |
| XGBoost | 0.8039 |
