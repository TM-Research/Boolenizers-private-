# MedicalImages

`time_series` · 10 classes · 798 train / 343 test · 99 features · imbalance 25.9×

**Winner: ML** — TM 0.773 (MovingWindowBinarizer) vs ML 0.785 (ExtraTrees), Δ -0.011


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | MovingWindowBinarizer | 0.7735 |
| 2 | DynamicPulseBinarizer | 0.7553 |
| 3 | GLADEBooleanizer | 0.7544 |
| 4 | SingleSpeedP2 | 0.7529 |
| 5 | DualDynamicsBinarizer | 0.7502 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| ExtraTrees | 0.7847 |
| RandomForest | 0.7583 |
| LightGBM | 0.7515 |
| XGBoost | 0.7016 |
