# FiftyWords

`time_series` · 50 classes · 633 train / 272 test · 270 features · imbalance 19.0×

**Winner: ML** — TM 0.545 (TWINELite) vs ML 0.599 (ExtraTrees), Δ -0.053


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | TWINELite | 0.5452 |
| 2 | SpectralStabilityBinarizer | 0.5451 |
| 3 | SignalQuantileFusion | 0.5448 |
| 4 | QBEspresso | 0.5365 |
| 5 | StandardBinarizerWrapper | 0.5322 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| ExtraTrees | 0.5985 |
| RandomForest | 0.5539 |
| XGBoost | 0.4323 |
| LightGBM | 0.0698 |
