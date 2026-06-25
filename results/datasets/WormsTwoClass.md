# WormsTwoClass

`time_series` · 2 classes · 180 train / 78 test · 512 features · imbalance 1.4×

**Winner: TM** — TM 0.613 (TWINEv3) vs ML 0.545 (XGBoost), Δ +0.068


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | TWINEv3 | 0.6131 |
| 2 | DriftRobustBinarizer | 0.5993 |
| 3 | PulseResonanceBinarizer | 0.5945 |
| 4 | SpectralStabilityBinarizer | 0.5887 |
| 5 | StandardBinarizerNative | 0.5887 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| XGBoost | 0.5452 |
| ExtraTrees | 0.5249 |
| LightGBM | 0.5099 |
| RandomForest | 0.4959 |
