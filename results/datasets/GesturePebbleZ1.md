# GesturePebbleZ1

`time_series` · 6 classes · 212 train / 92 test · 455 features · imbalance 1.2×

**Winner: TM** — TM 0.792 (PulseResonanceBinarizer) vs ML 0.792 (ExtraTrees), Δ +0.000


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | PulseResonanceBinarizer | 0.7919 |
| 2 | OnlineGeneralizedBinarizer | 0.7824 |
| 3 | QBEspresso | 0.7818 |
| 4 | ResonantGradientBinarizerV2 | 0.7812 |
| 5 | TWINEv2 | 0.7775 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| ExtraTrees | 0.7917 |
| RandomForest | 0.7693 |
| XGBoost | 0.7407 |
| LightGBM | 0.7374 |
