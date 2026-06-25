# ShapesAll

`time_series` · 60 classes · 840 train / 360 test · 512 features · imbalance 1.0×

**Winner: ML** — TM 0.721 (StandardBinarizerNative) vs ML 0.768 (ExtraTrees), Δ -0.047


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | StandardBinarizerNative | 0.7209 |
| 2 | AdaptiveQuantileBinarizer | 0.7168 |
| 3 | TWINEv3 | 0.7130 |
| 4 | AdaptiveMomentumBinarizer | 0.7114 |
| 5 | DecisionTreeBinarizer | 0.7100 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| ExtraTrees | 0.7681 |
| RandomForest | 0.7320 |
| XGBoost | 0.6390 |
| LightGBM | 0.0528 |
