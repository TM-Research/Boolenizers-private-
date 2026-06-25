# ChlorineConcentration

`time_series` · 3 classes · 3014 train / 1293 test · 166 features · imbalance 2.3×

**Winner: ML** — TM 0.994 (StandardBinarizerNative) vs ML 0.996 (LightGBM), Δ -0.001


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | StandardBinarizerNative | 0.9944 |
| 2 | GLADEBooleanizer | 0.9828 |
| 3 | GLADEEncoder | 0.9812 |
| 4 | OnlineGeneralizedBinarizer | 0.9812 |
| 5 | AdaptiveQuantileBinarizer | 0.9787 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| LightGBM | 0.9957 |
| XGBoost | 0.9917 |
| ExtraTrees | 0.9917 |
| RandomForest | 0.9857 |
