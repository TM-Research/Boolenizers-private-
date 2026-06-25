# NonInvasiveFetalECGThorax2

`time_series` · 42 classes · 2635 train / 1130 test · 512 features · imbalance 1.3×

**Winner: TM** — TM 0.936 (QBEspresso) vs ML 0.920 (ExtraTrees), Δ +0.016


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | QBEspresso | 0.9362 |
| 2 | DriftRobustBinarizer | 0.9355 |
| 3 | SAQT | 0.9349 |
| 4 | DualDynamicsBinarizer | 0.9333 |
| 5 | KBinsThermometer | 0.9333 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| ExtraTrees | 0.9205 |
| RandomForest | 0.9188 |
| XGBoost | 0.8982 |
| LightGBM | 0.1594 |
