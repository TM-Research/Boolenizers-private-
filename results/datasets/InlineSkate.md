# InlineSkate

`time_series` · 7 classes · 455 train / 195 test · 512 features · imbalance 1.9×

**Winner: ML** — TM 0.546 (MovingWindowBinarizerV2) vs ML 0.569 (ExtraTrees), Δ -0.023


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | MovingWindowBinarizerV2 | 0.5460 |
| 2 | NTEBatchQuantile | 0.5402 |
| 3 | SpectralStabilityBinarizer | 0.5365 |
| 4 | TWINEv3 | 0.5365 |
| 5 | SketchGK | 0.5345 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| ExtraTrees | 0.5686 |
| RandomForest | 0.5238 |
| XGBoost | 0.4820 |
| LightGBM | 0.4661 |
