# hynetsys

`tabular` · 3 classes · 30001 train / 10000 test · 63 features · imbalance 1.0×

**Winner: ML** — TM 0.400 (GLADEBooleanizer) vs ML 0.445 (XGBoost), Δ -0.045


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | GLADEBooleanizer | 0.3996 |
| 2 | ResonantGradientBinarizerV2 | 0.3986 |
| 3 | QBEspresso | 0.3944 |
| 4 | GLADEEncoder | 0.3943 |
| 5 | KalmanFilterBinarizer | 0.3921 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| XGBoost | 0.4451 |
| LightGBM | 0.4350 |
| RandomForest | 0.4239 |
| ExtraTrees | 0.4225 |
