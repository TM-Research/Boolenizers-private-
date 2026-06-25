# ProximalPhalanxTW

`time_series` · 6 classes · 423 train / 182 test · 80 features · imbalance 13.5×

**Winner: TM** — TM 0.618 (QBEspresso) vs ML 0.611 (LightGBM), Δ +0.006


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | QBEspresso | 0.6177 |
| 2 | SAQT | 0.5962 |
| 3 | OnlineGeneralizedBinarizer | 0.5832 |
| 4 | KBinsThermometer | 0.5778 |
| 5 | GLADEBooleanizer | 0.5748 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| LightGBM | 0.6114 |
| XGBoost | 0.5871 |
| RandomForest | 0.5432 |
| ExtraTrees | 0.4857 |
