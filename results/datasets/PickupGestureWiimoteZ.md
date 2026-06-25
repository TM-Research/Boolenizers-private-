# PickupGestureWiimoteZ

`time_series` · 10 classes · 70 train / 30 test · 361 features · imbalance 1.0×

**Winner: ML** — TM 0.489 (KnownMethodsBinarizer) vs ML 0.538 (ExtraTrees), Δ -0.050


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | KnownMethodsBinarizer | 0.4888 |
| 2 | DualDynamicsBinarizer | 0.4706 |
| 3 | StandardBinarizerNative | 0.4603 |
| 4 | OnlineQuantileTrackerBinarizer | 0.4205 |
| 5 | TWINELite | 0.4095 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| ExtraTrees | 0.5383 |
| RandomForest | 0.4854 |
| LightGBM | 0.4849 |
| XGBoost | 0.3843 |
