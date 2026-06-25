# WordSynonyms

`time_series` · 25 classes · 633 train / 272 test · 270 features · imbalance 17.5×

**Winner: ML** — TM 0.581 (OnlineQuantileTrackerBinarizer) vs ML 0.650 (ExtraTrees), Δ -0.069


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | OnlineQuantileTrackerBinarizer | 0.5808 |
| 2 | PulseResonanceBinarizer | 0.5599 |
| 3 | SpectralStabilityBinarizer | 0.5578 |
| 4 | NTEUniform | 0.5561 |
| 5 | KnownMethodsBinarizer | 0.5549 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| ExtraTrees | 0.6499 |
| RandomForest | 0.5664 |
| LightGBM | 0.5232 |
| XGBoost | 0.4770 |
