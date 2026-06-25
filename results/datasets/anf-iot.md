# anf-iot

`tabular` · 3 classes · 29999 train / 10000 test · 75 features · imbalance 1.8×

**Winner: ML** — TM 0.824 (GLADEEncoder) vs ML 0.843 (XGBoost), Δ -0.019


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | GLADEEncoder | 0.8239 |
| 2 | SAQT | 0.8238 |
| 3 | StandardBinarizerNative | 0.8230 |
| 4 | StandardBinarizerWrapper | 0.8225 |
| 5 | KnownMethodsBinarizer | 0.8200 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| XGBoost | 0.8427 |
| LightGBM | 0.8395 |
| RandomForest | 0.8365 |
| ExtraTrees | 0.8272 |
