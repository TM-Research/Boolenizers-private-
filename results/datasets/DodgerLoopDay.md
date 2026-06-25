# DodgerLoopDay

`time_series` · 7 classes · 100 train / 44 test · 288 features · imbalance 1.3×

**Winner: ML** — TM 0.620 (KnownMethodsBinarizer) vs ML 0.623 (ExtraTrees), Δ -0.002


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | KnownMethodsBinarizer | 0.6202 |
| 2 | DynamicPulseBinarizer | 0.5953 |
| 3 | ACFB | 0.5890 |
| 4 | OnlineBollingerBinarizer | 0.5726 |
| 5 | NTEUniform | 0.5658 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| ExtraTrees | 0.6226 |
| LightGBM | 0.6095 |
| RandomForest | 0.5609 |
| XGBoost | 0.3585 |
