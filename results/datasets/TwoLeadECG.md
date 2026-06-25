# TwoLeadECG

`time_series` · 2 classes · 813 train / 349 test · 82 features · imbalance 1.0×

**Winner: TM** — TM 0.997 (GLADEEncoder) vs ML 0.997 (LightGBM), Δ +0.000


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | GLADEEncoder | 0.9971 |
| 2 | NTEBatchQuantile | 0.9943 |
| 3 | OGBFast | 0.9943 |
| 4 | OnlineBollingerBinarizer | 0.9943 |
| 5 | PulseResonanceBinarizer | 0.9943 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| LightGBM | 0.9971 |
| ExtraTrees | 0.9857 |
| XGBoost | 0.9713 |
| RandomForest | 0.9627 |
