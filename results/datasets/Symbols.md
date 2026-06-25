# Symbols

`time_series` · 6 classes · 714 train / 306 test · 398 features · imbalance 1.1×

**Winner: ML** — TM 0.977 (DecisionTreeBinarizer) vs ML 0.977 (ExtraTrees), Δ -0.000


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | DecisionTreeBinarizer | 0.9773 |
| 2 | DualDynamicsBinarizer | 0.9741 |
| 3 | ResonantGradientBinarizer | 0.9741 |
| 4 | TWINEv2 | 0.9741 |
| 5 | ACFB | 0.9709 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| ExtraTrees | 0.9774 |
| RandomForest | 0.9741 |
| XGBoost | 0.9644 |
| LightGBM | 0.9641 |
