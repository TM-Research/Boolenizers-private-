# GunPointOldVersusYoung

`time_series` · 2 classes · 315 train / 136 test · 150 features · imbalance 1.1×

**Winner: ML** — TM 0.963 (MovingWindowBinarizerV2) vs ML 0.971 (ExtraTrees), Δ -0.007


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | MovingWindowBinarizerV2 | 0.9631 |
| 2 | NTEUniform | 0.9556 |
| 3 | ResonantGradientBinarizer | 0.9485 |
| 4 | StandardBinarizerWrapper | 0.9485 |
| 5 | QBEspresso | 0.9484 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| ExtraTrees | 0.9705 |
| XGBoost | 0.9556 |
| LightGBM | 0.9482 |
| RandomForest | 0.9410 |
