# nids-bench-2026

`tabular` · 13 classes · 30000 train / 10000 test · 62 features · imbalance 26.0×

**Winner: TM** — TM 0.869 (AQB) vs ML 0.867 (XGBoost), Δ +0.001


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | AQB | 0.8685 |
| 2 | MWAB | 0.8680 |
| 3 | ResonantGradientBinarizerV2 | 0.8638 |
| 4 | OnlineGeneralizedBinarizer | 0.8637 |
| 5 | DualDynamicsBinarizer | 0.8635 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| XGBoost | 0.8673 |
| LightGBM | 0.8665 |
| RandomForest | 0.8613 |
| ExtraTrees | 0.8443 |
