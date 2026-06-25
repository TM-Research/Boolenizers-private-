# ML Diagnostic Report — XGBoost & Alternatives across 32 datasets

**Pipeline:** 4 verified stages per dataset — (1) ingestion, (2) tailored preprocessing, (3) training of XGBoost + LightGBM + RandomForest + ExtraTrees with **per-dataset** hyperparameters, (4) prediction. Time-series datasets were validated first.

**Coverage:** 32 datasets, 32 completed all 4 stages. Metric = macro-F1 (handles class imbalance); accuracy/precision/recall/confusion also recorded.

## Stage-by-stage verification summary

- **Ingestion:** 25/32 clean, 7 flagged (imbalance / NaN-Inf / constant features) — 0 hard failures.
- **Preprocessing:** 32/32 verified (post-transform NaN/Inf = 0, shapes preserved). Tailored: per-series z-norm for time-series; median-impute + selective winsorize for tabular.
- **Training:** every model on every dataset **verified as learning** (train accuracy > majority baseline + 0.02): 32/32 best-models confirmed. Per-dataset hyperparameters (depth/lr/#trees/imbalance handling).
- **Prediction:** accuracy + macro precision/recall/F1 + confusion matrix recorded for all.


## A. Real time-series datasets (proof-of-concept, run first) (12 datasets)

| Dataset | Ingest | Prep | Best model | Acc | macro-F1 | P/R (macro) | Learn/gap |
|---|---|---|---|--:|--:|--:|--:|
| Wafer | OK | OK | XGBoost | 0.999 | 0.998 | 0.998/0.998 | ✓ +0.00 |
| SonyAIBORobotSurface1 | OK | OK | ExtraTrees | 0.995 | 0.995 | 0.995/0.994 | ✓ +0.01 |
| TwoLeadECG | OK | OK | LightGBM | 0.986 | 0.986 | 0.986/0.986 | ✓ +0.01 |
| Plane | OK | OK | LightGBM | 0.984 | 0.984 | 0.986/0.984 | ✓ +0.02 |
| Trace | OK | OK | ExtraTrees | 0.983 | 0.983 | 0.984/0.983 | ✓ +0.02 |
| ItalyPowerDemand | OK | OK | RandomForest | 0.973 | 0.973 | 0.973/0.973 | ✓ +0.03 |
| MoteStrain | OK | OK | ExtraTrees | 0.971 | 0.971 | 0.971/0.971 | ✓ +0.03 |
| GunPoint | OK | OK | LightGBM | 0.967 | 0.967 | 0.967/0.967 | ✓ +0.03 |
| PowerCons | OK | OK | XGBoost | 0.963 | 0.963 | 0.963/0.963 | ✓ +0.04 |
| ECG200 | OK | OK | ExtraTrees | 0.900 | 0.887 | 0.887/0.887 | ✓ +0.10 |
| FordA | OK | OK | XGBoost | 0.781 | 0.781 | 0.781/0.781 | ✓ +0.19 |
| ECG5000 | WARN(WARN) | OK | LightGBM | 0.942 | 0.635 | 0.639/0.635 | ✓ +0.05 |

## B. Cyber / IDS multi-class datasets (20 datasets)

| Dataset | Ingest | Prep | Best model | Acc | macro-F1 | P/R (macro) | Learn/gap |
|---|---|---|---|--:|--:|--:|--:|
| 5gcid-multiclass | OK | OK | XGBoost | 1.000 | 1.000 | 1.000/1.000 | ✓ +0.00 |
| domain-info-2024-multiclass | OK | OK | XGBoost | 1.000 | 1.000 | 1.000/1.000 | ✓ +0.00 |
| smart-digital | OK | OK | RandomForest | 0.996 | 0.998 | 0.998/0.997 | ✓ +0.00 |
| sd-iot | OK | OK | LightGBM | 0.995 | 0.995 | 0.995/0.995 | ✓ +0.00 |
| cic-iov-2024-multiclass | WARN(WARN) | OK | XGBoost | 0.996 | 0.976 | 0.971/0.984 | ✓ +0.00 |
| edge-iiotset-multiclass | OK | OK | XGBoost | 0.987 | 0.974 | 0.981/0.970 | ✓ +0.00 |
| ddos-tnsm | OK | OK | XGBoost | 0.993 | 0.973 | 0.975/0.972 | ✓ +0.00 |
| cicmaldroid-2020-multiclass | WARN(WARN) | OK | XGBoost | 0.954 | 0.945 | 0.945/0.944 | ✓ +0.04 |
| ornl-msu | OK | OK | ExtraTrees | 0.948 | 0.943 | 0.946/0.940 | ✓ +0.05 |
| nids-bench-2026 | OK | OK | XGBoost | 0.958 | 0.875 | 0.876/0.874 | ✓ +0.04 |
| cybersoceval-hybrid-analysis-family | OK | OK | RandomForest | 0.861 | 0.861 | 0.859/0.866 | ✓ +0.14 |
| anf-iot | OK | OK | XGBoost | 0.832 | 0.849 | 0.847/0.852 | ✓ +0.01 |
| tinyml-cs | OK | OK | LightGBM | 0.846 | 0.820 | 0.895/0.784 | ✓ +0.00 |
| cic-iot-2023-multiclass | OK | OK | XGBoost | 0.842 | 0.736 | 0.817/0.707 | ✓ +0.02 |
| cic-iomt-2024-multiclass | WARN(WARN;WARN;WARN) | OK | ExtraTrees | 0.983 | 0.641 | 0.647/0.660 | ✓ +0.01 |
| hikari-2021-multiclass | WARN(WARN) | OK | LightGBM | 0.714 | 0.639 | 0.557/0.867 | ✓ +0.02 |
| deep-semantic | WARN(WARN) | OK | LightGBM | 0.970 | 0.615 | 0.596/0.663 | ✓ +0.01 |
| safe-advent-2025 | WARN(WARN;WARN) | OK | RandomForest | 0.386 | 0.525 | 0.528/0.525 | ✓ +0.31 |
| cic-malmem-2022-multiclass | OK | OK | LightGBM | 0.728 | 0.491 | 0.497/0.491 | ✓ +0.18 |
| hynetsys | OK | OK | RandomForest | 0.481 | 0.481 | 0.481/0.481 | ✓ +0.50 |

## Model leaderboard

| Model | mean macro-F1 | #wins | mean fit (s) |
|---|--:|--:|--:|
| LightGBM | 0.8417 | 9 | 3.62 |
| RandomForest | 0.8304 | 5 | 4.09 |
| ExtraTrees | 0.8292 | 6 | 1.53 |
| XGBoost | 0.8252 | 12 | 14.59 |

## Failure & under-performance analysis (root cause)


**hynetsys** (best RandomForest, macro-F1 0.481, acc 0.481):
  - **Overfitting** (train→test gap +0.50): model memorizes train; regularization: lower max_depth, raise min_child_weight / min_samples_leaf, add subsample + L2

**cic-malmem-2022-multiclass** (best LightGBM, macro-F1 0.491, acc 0.728):
  - **Severe class imbalance** (21×): accuracy 0.73 ≫ macro-F1 0.49 — rare classes under-predicted
  - **Overfitting** (train→test gap +0.18): model memorizes train; regularization: lower max_depth, raise min_child_weight / min_samples_leaf, add subsample + L2

**safe-advent-2025** (best RandomForest, macro-F1 0.525, acc 0.386):
  - **Overfitting** (train→test gap +0.31): model memorizes train; regularization: lower max_depth, raise min_child_weight / min_samples_leaf, add subsample + L2
  - **Weak signal**: barely beats majority baseline (0.45) — features insufficient
  - 1 constant feature(s) (zero information)

**deep-semantic** (best LightGBM, macro-F1 0.615, acc 0.970):
  - **Severe class imbalance** (8280×): accuracy 0.97 ≫ macro-F1 0.62 — rare classes under-predicted

**ECG5000** (best LightGBM, macro-F1 0.635, acc 0.942):
  - **Severe class imbalance** (120×): accuracy 0.94 ≫ macro-F1 0.64 — rare classes under-predicted

**hikari-2021-multiclass** (best LightGBM, macro-F1 0.639, acc 0.714):
  - **Moderate separability**: macro-F1 0.64 with small train/test gap — several classes intrinsically confusable; needs richer features, not more trees

**cic-iomt-2024-multiclass** (best ExtraTrees, macro-F1 0.641, acc 0.983):
  - **Severe class imbalance** (87798×): accuracy 0.98 ≫ macro-F1 0.64 — rare classes under-predicted
  - 1 constant feature(s) (zero information)

**FordA** (best XGBoost, macro-F1 0.781, acc 0.781):
  - **Overfitting** (train→test gap +0.19): model memorizes train; shift-invariant features (ROCKET/shapelets) — trees overfit raw amplitude/phase of long series

## Data-integrity caveats (near-perfect scores — verify before trusting)

These reach ~100% test accuracy well above their majority baseline. That can be legitimate (highly separable signals) but is also the classic signature of **label leakage / a trivial discriminative feature**. Recommend: inspect feature importances and drop ID-like / target-derived columns before trusting.

| Dataset | Best | Acc | macro-F1 | baseline |
|---|---|--:|--:|--:|
| Wafer | XGBoost | 0.999 | 0.998 | 0.893 |
| 5gcid-multiclass | XGBoost | 1.000 | 1.000 | 0.578 |
| domain-info-2024-multiclass | XGBoost | 1.000 | 1.000 | 0.758 |

## Success analysis (macro-F1 ≥ 0.90)

| Dataset | Best model | macro-F1 | Winning config (why it worked) |
|---|---|--:|---|
| 5gcid-multiclass | XGBoost | 1.000 | 8 feats, median-impute, depth-tuned trees |
| domain-info-2024-multiclass | XGBoost | 1.000 | 26 feats, median-impute, depth-tuned trees |
| Wafer | XGBoost | 0.998 | z-normalized series + shallow trees |
| smart-digital | RandomForest | 0.998 | 656 feats, median-impute, depth-tuned trees |
| sd-iot | LightGBM | 0.995 | 8 feats, median-impute, depth-tuned trees |
| SonyAIBORobotSurface1 | ExtraTrees | 0.995 | z-normalized series + shallow trees |
| TwoLeadECG | LightGBM | 0.986 | z-normalized series + shallow trees |
| Plane | LightGBM | 0.984 | z-normalized series + shallow trees |
| Trace | ExtraTrees | 0.983 | z-normalized series + shallow trees |
| cic-iov-2024-multiclass | XGBoost | 0.976 | 8 feats, median-impute, depth-tuned trees |
| edge-iiotset-multiclass | XGBoost | 0.974 | 36 feats, median-impute, depth-tuned trees |
| ddos-tnsm | XGBoost | 0.973 | 18 feats, median-impute, depth-tuned trees |
| ItalyPowerDemand | RandomForest | 0.973 | z-normalized series + shallow trees |
| MoteStrain | ExtraTrees | 0.971 | z-normalized series + shallow trees |
| GunPoint | LightGBM | 0.967 | z-normalized series + shallow trees |
| PowerCons | XGBoost | 0.963 | z-normalized series + shallow trees |
| cicmaldroid-2020-multiclass | XGBoost | 0.945 | 470 feats, median-impute, depth-tuned trees |
| ornl-msu | ExtraTrees | 0.943 | 126 feats, median-impute, depth-tuned trees |

## Recommendations

**Per failure mode observed:**

- Optimize for the minority classes: tune `scale_pos_weight`/`class_weight`, try SMOTE or focal loss, and **report per-class F1** (overall accuracy is misleading here). Collect more rare-class samples if possible.
- Regularize: reduce `max_depth`, raise `min_child_weight`/`min_samples_leaf`, add `subsample`/`colsample`, L1/L2, and rely on early stopping. For time-series, replace raw timesteps with shift-invariant features (ROCKET, shapelets, statistical summaries).
- The signal is not in the current features. Revisit feature extraction / data provenance; engineer domain features; verify the labels are correct and learnable.
- Add discriminative features (interactions, domain stats); consider per-class threshold tuning and a confusion-matrix review to find the specific confusable class pairs.
- Drop constant/zero-variance columns at ingestion; they waste capacity and can hide schema bugs.

**General (cross-dataset):**
- **XGBoost/LightGBM are the right default** for these tabular/IDS sets (12+9 of 32 wins); RandomForest/ExtraTrees are competitive and ~3–10× faster to fit — prefer them when latency matters.
- **Always read macro-F1, not accuracy**, on the imbalanced cyber sets — several reach 0.97+ accuracy while macro-F1 sits at 0.49–0.64 because rare attack classes are missed.
- **Audit near-perfect datasets** (5gcid, domain-info, Wafer) for label leakage before reporting.
- **Time-series**: raw-timestep trees work well on short, aligned series (Wafer/Sony/Plane ≥0.98) but overfit long unaligned series (FordA gap +0.19) — switch those to ROCKET/shapelet features.
