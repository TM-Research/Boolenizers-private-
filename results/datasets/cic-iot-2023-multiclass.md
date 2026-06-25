# cic-iot-2023-multiclass

`tabular` · 8 classes · 30001 train / 10001 test · 39 features · imbalance 34.6×

**Winner: ML** — TM 0.643 (ResonantGradientBinarizerV2) vs ML 0.709 (XGBoost), Δ -0.066


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | ResonantGradientBinarizerV2 | 0.6428 |
| 2 | OQSB | 0.6386 |
| 3 | MovingWindowBinarizer | 0.6384 |
| 4 | GLADEBooleanizer | 0.6332 |
| 5 | AQB | 0.6256 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| XGBoost | 0.7090 |
| LightGBM | 0.7048 |
| RandomForest | 0.6964 |
| ExtraTrees | 0.6828 |
