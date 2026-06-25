# cic-iomt-2024-multiclass

`tabular` · 19 classes · 30001 train / 10004 test · 73 features · imbalance 13170.0×

**Winner: ML** — TM 0.553 (OnlineGeneralizedBinarizer) vs ML 0.557 (XGBoost), Δ -0.004


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | OnlineGeneralizedBinarizer | 0.5533 |
| 2 | MovingWindowBinarizer | 0.5460 |
| 3 | SDQB | 0.5367 |
| 4 | AdaptiveMomentumBinarizer | 0.5336 |
| 5 | PulseResonanceBinarizer | 0.5311 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| XGBoost | 0.5568 |
| LightGBM | 0.5381 |
| RandomForest | 0.4957 |
| ExtraTrees | 0.4925 |
