# NonInvasiveFetalECGThorax1

`time_series` · 42 classes · 2635 train / 1130 test · 512 features · imbalance 1.3×

**Winner: TM** — TM 0.927 (PulseResonanceBinarizer) vs ML 0.894 (ExtraTrees), Δ +0.032


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | PulseResonanceBinarizer | 0.9268 |
| 2 | SAQT | 0.9250 |
| 3 | OnlineGeneralizedBinarizer | 0.9239 |
| 4 | SignalQuantileFusion | 0.9239 |
| 5 | ResonantGradientBinarizerV2 | 0.9219 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| ExtraTrees | 0.8944 |
| RandomForest | 0.8865 |
| XGBoost | 0.8497 |
| LightGBM | 0.1203 |
