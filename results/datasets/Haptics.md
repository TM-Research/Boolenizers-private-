# Haptics

`time_series` · 5 classes · 324 train / 139 test · 512 features · imbalance 1.3×

**Winner: TM** — TM 0.520 (PulseResonanceBinarizer) vs ML 0.494 (XGBoost), Δ +0.026


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | PulseResonanceBinarizer | 0.5199 |
| 2 | DecisionTreeBinarizer | 0.5088 |
| 3 | TWINELite | 0.5071 |
| 4 | OnlineBollingerBinarizer | 0.5020 |
| 5 | NTEUniform | 0.4999 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| XGBoost | 0.4936 |
| ExtraTrees | 0.4721 |
| LightGBM | 0.4543 |
| RandomForest | 0.4470 |
