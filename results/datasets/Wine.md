# Wine

`time_series` · 2 classes · 77 train / 34 test · 234 features · imbalance 1.1×

**Winner: ML** — TM 0.941 (GLADEEncoder) vs ML 0.941 (LightGBM), Δ -0.000


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | GLADEEncoder | 0.9410 |
| 2 | KnownMethodsBinarizer | 0.9410 |
| 3 | MovingWindowBinarizer | 0.9410 |
| 4 | OnlineBollingerBinarizer | 0.9410 |
| 5 | OnlineQuantileTrackerBinarizer | 0.9410 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| LightGBM | 0.9412 |
| XGBoost | 0.9117 |
| RandomForest | 0.9117 |
| ExtraTrees | 0.8824 |
