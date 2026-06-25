# TwoPatterns

`time_series` · 4 classes · 3500 train / 1500 test · 128 features · imbalance 1.1×

**Winner: ML** — TM 0.977 (NTEUniform) vs ML 0.991 (ExtraTrees), Δ -0.014


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | NTEUniform | 0.9766 |
| 2 | OGBFast | 0.9754 |
| 3 | ACFB | 0.9748 |
| 4 | OnlineBollingerBinarizer | 0.9679 |
| 5 | ResonantGradientBinarizerV2 | 0.9655 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| ExtraTrees | 0.9905 |
| RandomForest | 0.9771 |
| LightGBM | 0.9761 |
| XGBoost | 0.9634 |
