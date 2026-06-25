# CricketX

`time_series` · 12 classes · 546 train / 234 test · 300 features · imbalance 1.0×

**Winner: TM** — TM 0.719 (NTEBatchQuantile) vs ML 0.694 (ExtraTrees), Δ +0.025


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | NTEBatchQuantile | 0.7192 |
| 2 | AdaptiveQuantileBinarizer | 0.7129 |
| 3 | OnlineGeneralizedBinarizer | 0.7080 |
| 4 | SketchGK | 0.7076 |
| 5 | MovingWindowBinarizerV2 | 0.7072 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| ExtraTrees | 0.6943 |
| RandomForest | 0.6735 |
| LightGBM | 0.6378 |
| XGBoost | 0.6260 |
