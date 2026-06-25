# FordA

`time_series` · 2 classes · 3444 train / 1477 test · 500 features · imbalance 1.1×

**Winner: ML** — TM 0.812 (OnlineQuantileTrackerBinarizer) vs ML 0.823 (LightGBM), Δ -0.011


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | OnlineQuantileTrackerBinarizer | 0.8124 |
| 2 | ResonantGradientBinarizer | 0.8101 |
| 3 | SketchGK | 0.8076 |
| 4 | GLADEBooleanizer | 0.8070 |
| 5 | GLADEEncoder | 0.8057 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| LightGBM | 0.8232 |
| XGBoost | 0.8050 |
| ExtraTrees | 0.7612 |
| RandomForest | 0.7439 |
