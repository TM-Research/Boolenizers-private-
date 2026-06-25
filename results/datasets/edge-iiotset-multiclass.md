# edge-iiotset-multiclass

`tabular` · 15 classes · 30000 train / 9999 test · 36 features · imbalance 24.3×

**Winner: ML** — TM 0.908 (OnlineGeneralizedBinarizer) vs ML 0.974 (XGBoost), Δ -0.065


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | OnlineGeneralizedBinarizer | 0.9085 |
| 2 | StandardBinarizerNative | 0.9012 |
| 3 | ResonantGradientBinarizerV2 | 0.8982 |
| 4 | GLADEEncoder | 0.8912 |
| 5 | GLADEBooleanizer | 0.8910 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| XGBoost | 0.9739 |
| RandomForest | 0.9590 |
| LightGBM | 0.9577 |
| ExtraTrees | 0.9548 |
