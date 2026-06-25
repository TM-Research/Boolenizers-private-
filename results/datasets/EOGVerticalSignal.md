# EOGVerticalSignal

`time_series` · 12 classes · 506 train / 218 test · 512 features · imbalance 1.0×

**Winner: ML** — TM 0.618 (NTEUniform) vs ML 0.625 (ExtraTrees), Δ -0.007


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | NTEUniform | 0.6180 |
| 2 | TWINEv2 | 0.6111 |
| 3 | DecisionTreeBinarizer | 0.6072 |
| 4 | SpectralStabilityBinarizer | 0.6064 |
| 5 | MovingWindowBinarizer | 0.6060 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| ExtraTrees | 0.6250 |
| LightGBM | 0.5962 |
| RandomForest | 0.5783 |
| XGBoost | 0.4987 |
