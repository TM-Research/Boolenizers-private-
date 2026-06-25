# ddos-tnsm

`tabular` · 7 classes · 29999 train / 10001 test · 18 features · imbalance 14.5×

**Winner: ML** — TM 0.963 (AdaptiveMomentumBinarizer) vs ML 0.965 (LightGBM), Δ -0.002


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | AdaptiveMomentumBinarizer | 0.9630 |
| 2 | DualDynamicsBinarizer | 0.9599 |
| 3 | StandardBinarizerNative | 0.9587 |
| 4 | KnownMethodsBinarizer | 0.9574 |
| 5 | AdaptiveQuantileBinarizer | 0.9570 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| LightGBM | 0.9654 |
| XGBoost | 0.9625 |
| RandomForest | 0.9584 |
| ExtraTrees | 0.9583 |
