# PLAID

`time_series` · 11 classes · 751 train / 323 test · 512 features · imbalance 6.8×

**Winner: TM** — TM 0.599 (GLADEBooleanizer) vs ML 0.528 (RandomForest), Δ +0.072


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | GLADEBooleanizer | 0.5994 |
| 2 | GLADEEncoder | 0.5957 |
| 3 | DecisionTreeBinarizer | 0.5825 |
| 4 | KBinsThermometer | 0.5799 |
| 5 | TWINELite | 0.5781 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| RandomForest | 0.5278 |
| LightGBM | 0.5133 |
| XGBoost | 0.4951 |
| ExtraTrees | 0.4490 |
