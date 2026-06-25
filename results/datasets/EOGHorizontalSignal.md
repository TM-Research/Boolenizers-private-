# EOGHorizontalSignal

`time_series` · 12 classes · 506 train / 218 test · 512 features · imbalance 1.0×

**Winner: TM** — TM 0.723 (ACFB) vs ML 0.687 (ExtraTrees), Δ +0.036


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | ACFB | 0.7230 |
| 2 | DualDynamicsBinarizer | 0.6876 |
| 3 | NTEUniform | 0.6829 |
| 4 | PulseResonanceBinarizer | 0.6783 |
| 5 | OnlineGeneralizedBinarizer | 0.6771 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| ExtraTrees | 0.6870 |
| RandomForest | 0.6519 |
| LightGBM | 0.6440 |
| XGBoost | 0.6176 |
