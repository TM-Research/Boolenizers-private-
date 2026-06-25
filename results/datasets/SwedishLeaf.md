# SwedishLeaf

`time_series` · 15 classes · 787 train / 338 test · 128 features · imbalance 1.0×

**Winner: TM** — TM 0.923 (NTEUniform) vs ML 0.905 (ExtraTrees), Δ +0.018


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | NTEUniform | 0.9233 |
| 2 | AdaptiveQuantileBinarizer | 0.9202 |
| 3 | SpectralStabilityBinarizer | 0.9200 |
| 4 | GLADEBooleanizer | 0.9196 |
| 5 | SAQT | 0.9173 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| ExtraTrees | 0.9049 |
| RandomForest | 0.8990 |
| LightGBM | 0.8885 |
| XGBoost | 0.8497 |
