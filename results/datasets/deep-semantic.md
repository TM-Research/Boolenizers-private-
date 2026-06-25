# deep-semantic

`tabular` · 10 classes · 29999 train / 9999 test · 47 features · imbalance 9108.3×

**Winner: ML** — TM 0.464 (KBinsThermometer) vs ML 0.465 (XGBoost), Δ -0.000


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | KBinsThermometer | 0.4643 |
| 2 | StandardBinarizerNative | 0.4602 |
| 3 | SDQB | 0.4596 |
| 4 | KnownMethodsBinarizer | 0.4524 |
| 5 | NTEBatchQuantile | 0.4521 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| XGBoost | 0.4645 |
| RandomForest | 0.4628 |
| LightGBM | 0.4589 |
| ExtraTrees | 0.4588 |
