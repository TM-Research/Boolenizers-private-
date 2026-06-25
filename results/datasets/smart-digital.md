# smart-digital

`tabular` · 10 classes · 2160 train / 540 test · 656 features · imbalance 3.7×

**Winner: ML** — TM 0.995 (KBinsThermometer) vs ML 0.998 (RandomForest), Δ -0.003


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | KBinsThermometer | 0.9950 |
| 2 | AdaptiveMomentumBinarizer | 0.9949 |
| 3 | DualDynamicsBinarizer | 0.9949 |
| 4 | KnownMethodsBinarizer | 0.9949 |
| 5 | QBEspresso | 0.9949 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| RandomForest | 0.9975 |
| ExtraTrees | 0.9949 |
| XGBoost | 0.9925 |
| LightGBM | 0.9925 |
