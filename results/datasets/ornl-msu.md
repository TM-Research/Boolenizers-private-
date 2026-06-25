# ornl-msu

`tabular` · 37 classes · 30000 train / 7185 test · 126 features · imbalance 4.0×

**Winner: ML** — TM 0.808 (StandardBinarizerNative) vs ML 0.896 (ExtraTrees), Δ -0.088


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | StandardBinarizerNative | 0.8081 |
| 2 | SAQT | 0.7890 |
| 3 | GLADEEncoder | 0.7847 |
| 4 | GLADEBooleanizer | 0.7781 |
| 5 | OnlineGeneralizedBinarizer | 0.7754 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| ExtraTrees | 0.8959 |
| LightGBM | 0.8646 |
| RandomForest | 0.8583 |
| XGBoost | 0.8542 |
