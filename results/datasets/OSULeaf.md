# OSULeaf

`time_series` · 6 classes · 309 train / 133 test · 427 features · imbalance 2.5×

**Winner: TM** — TM 0.707 (OnlineQuantileTrackerBinarizer) vs ML 0.680 (ExtraTrees), Δ +0.027


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | OnlineQuantileTrackerBinarizer | 0.7072 |
| 2 | KBinsThermometer | 0.6855 |
| 3 | StandardBinarizerWrapper | 0.6796 |
| 4 | SAQT | 0.6783 |
| 5 | ACFB | 0.6704 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| ExtraTrees | 0.6801 |
| LightGBM | 0.6323 |
| RandomForest | 0.6192 |
| XGBoost | 0.5902 |
