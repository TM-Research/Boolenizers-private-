# ProximalPhalanxOutlineAgeGroup

`time_series` · 3 classes · 423 train / 182 test · 80 features · imbalance 3.3×

**Winner: TM** — TM 0.829 (ResonantGradientBinarizer) vs ML 0.812 (ExtraTrees), Δ +0.017


## Top-5 booleanizers (TM macro-F1)

| # | Booleanizer | macro-F1 |
|--:|---|--:|
| 1 | ResonantGradientBinarizer | 0.8288 |
| 2 | DynamicPulseBinarizer | 0.8275 |
| 3 | TWINELite | 0.8257 |
| 4 | NTEUniform | 0.8188 |
| 5 | SpectralStabilityBinarizer | 0.8188 |

## ML models (macro-F1)

| Model | macro-F1 |
|---|--:|
| ExtraTrees | 0.8117 |
| RandomForest | 0.8111 |
| XGBoost | 0.8075 |
| LightGBM | 0.7813 |
