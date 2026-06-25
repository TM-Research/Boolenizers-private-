# cybersoceval-hybrid-analysis-family

`tabular` · 5 classes · 140 train / 36 test · 38 features · imbalance 1.5×

**Winner: TM** — TM 0.948 (DecisionTreeBinarizer) vs ML 0.857 (ExtraTrees), Δ +0.091


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | DecisionTreeBinarizer | 0.9478 |
| 2 | OGBFast | 0.9177 |
| 3 | ResonantGradientBinarizerV2 | 0.8614 |
| 4 | KnownMethodsBinarizer | 0.8539 |
| 5 | OnlineQuantileSignalBinarizer | 0.8391 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| ExtraTrees | 0.8566 |
| XGBoost | 0.8463 |
| LightGBM | 0.8449 |
| RandomForest | 0.8070 |
