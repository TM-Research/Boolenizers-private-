# MiddlePhalanxOutlineCorrect

`time_series` · 2 classes · 623 train / 268 test · 80 features · imbalance 1.6×

**Winner: TM** — TM 0.812 (GLADEBooleanizer) vs ML 0.796 (XGBoost), Δ +0.016


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | GLADEBooleanizer | 0.8116 |
| 2 | SketchGK | 0.8106 |
| 3 | OnlineQuantileTrackerBinarizer | 0.8087 |
| 4 | KnownMethodsBinarizer | 0.8079 |
| 5 | SignalGradientBinarizer | 0.8076 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| XGBoost | 0.7956 |
| LightGBM | 0.7801 |
| ExtraTrees | 0.7796 |
| RandomForest | 0.7532 |
