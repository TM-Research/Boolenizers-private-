# hikari-2021-multiclass

`tabular` · 6 classes · 30001 train / 10000 test · 81 features · imbalance 106.1×

**Winner: ML** — TM 0.576 (OnlineQuantileSignalBinarizer) vs ML 0.615 (LightGBM), Δ -0.040


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | OnlineQuantileSignalBinarizer | 0.5757 |
| 2 | OQSB | 0.5713 |
| 3 | OnlineDeltaMomentumBinarizer | 0.5681 |
| 4 | SketchGK | 0.5553 |
| 5 | OnlineUniversalBinarizer | 0.5543 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| LightGBM | 0.6152 |
| XGBoost | 0.5357 |
| ExtraTrees | 0.5140 |
| RandomForest | 0.5044 |
