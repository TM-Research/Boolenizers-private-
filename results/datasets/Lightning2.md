# Lightning2

`time_series` · 2 classes · 84 train / 37 test · 512 features · imbalance 1.5×

**Winner: ML** — TM 0.720 (DriftRobustBinarizer) vs ML 0.745 (XGBoost), Δ -0.025


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | DriftRobustBinarizer | 0.7197 |
| 2 | QBEspresso | 0.7197 |
| 3 | StandardBinarizerWrapper | 0.7197 |
| 4 | DynamicPulseBinarizer | 0.7127 |
| 5 | MWAB | 0.7127 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| XGBoost | 0.7448 |
| RandomForest | 0.7448 |
| ExtraTrees | 0.7376 |
| LightGBM | 0.6636 |
