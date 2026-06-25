# GesturePebbleZ2

`time_series` · 6 classes · 212 train / 92 test · 455 features · imbalance 1.2×

**Winner: TM** — TM 0.871 (GLADEEncoder) vs ML 0.845 (ExtraTrees), Δ +0.026


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | GLADEEncoder | 0.8709 |
| 2 | GLADEBooleanizer | 0.8700 |
| 3 | KBinsThermometer | 0.8688 |
| 4 | ResonantGradientBinarizerV2 | 0.8674 |
| 5 | TWINEv2 | 0.8606 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| ExtraTrees | 0.8447 |
| LightGBM | 0.8112 |
| RandomForest | 0.7999 |
| XGBoost | 0.7465 |
