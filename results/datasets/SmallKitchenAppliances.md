# SmallKitchenAppliances

`time_series` · 3 classes · 525 train / 225 test · 512 features · imbalance 1.0×

**Winner: TM** — TM 0.728 (ResonantGradientBinarizerV2) vs ML 0.723 (ExtraTrees), Δ +0.005


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | ResonantGradientBinarizerV2 | 0.7282 |
| 2 | StandardBinarizerNative | 0.7228 |
| 3 | SketchGK | 0.7210 |
| 4 | KBinsThermometer | 0.7201 |
| 5 | ResonantGradientBinarizer | 0.7201 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| ExtraTrees | 0.7228 |
| XGBoost | 0.6846 |
| LightGBM | 0.6720 |
| RandomForest | 0.6506 |
