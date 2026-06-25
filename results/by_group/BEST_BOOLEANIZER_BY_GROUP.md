# Best booleanizer by group (macro-F1)

Aggregated from the **top-5 booleanizers per dataset** in each group. Metric = macro-F1 (not accuracy).

**TL;DR — best booleanizer per group:**

| Group | datasets | Best booleanizer | top-5 | rank-1 wins | mean F1 |
|---|--:|---|--:|--:|--:|
| UCR time-series archive | 116 | **ACFB** | 34 (29%) | 14 | 0.736 |
| Original 12 time-series | 12 | **GLADEEncoder** | 5 (42%) | 2 | 0.924 |
| Cyber / IDS multi-class | 20 | **GLADEBooleanizer** | 8 (40%) | 2 | 0.738 |


## UCR time-series archive (116 datasets)

| Rank | Booleanizer | in top-5 | % | rank-1 wins | mean F1 |
|--:|---|--:|--:|--:|--:|
| 1 | ACFB ⭐ | 34 | 29% | 14 | 0.7362 |
| 2 | DecisionTreeBinarizer | 25 | 22% | 3 | 0.7349 |
| 3 | NTEUniform | 22 | 19% | 8 | 0.7364 |
| 4 | AdaptiveQuantileBinarizer | 20 | 17% | 4 | 0.7339 |
| 5 | DriftRobustBinarizer | 19 | 16% | 3 | 0.7337 |
| 6 | GLADEBooleanizer | 19 | 16% | 3 | 0.7297 |
| 7 | StandardBinarizerNative | 19 | 16% | 3 | 0.7343 |
| 8 | OnlineGeneralizedBinarizer | 19 | 16% | 3 | 0.7356 |
| 9 | SpectralStabilityBinarizer | 19 | 16% | 0 | 0.7364 |
| 10 | SAQT | 18 | 16% | 3 | 0.7318 |

## Original 12 time-series (12 datasets)

| Rank | Booleanizer | in top-5 | % | rank-1 wins | mean F1 |
|--:|---|--:|--:|--:|--:|
| 1 | GLADEEncoder ⭐ | 5 | 42% | 2 | 0.9241 |
| 2 | GLADEBooleanizer | 3 | 25% | 1 | 0.9185 |
| 3 | AdaptiveQuantileBinarizer | 3 | 25% | 2 | 0.9244 |
| 4 | DriftRobustBinarizer | 3 | 25% | 1 | 0.9211 |
| 5 | SAQT | 3 | 25% | 0 | 0.9271 |
| 6 | AdaptiveGaussian | 2 | 17% | 2 | 0.9032 |
| 7 | ACFB | 2 | 17% | 1 | 0.9250 |
| 8 | NTEBatchQuantile | 2 | 17% | 0 | 0.9252 |
| 9 | DecisionTreeBinarizer | 2 | 17% | 1 | 0.9184 |
| 10 | KalmanFilterBinarizer | 2 | 17% | 0 | 0.9208 |

## Cyber / IDS multi-class (20 datasets)

| Rank | Booleanizer | in top-5 | % | rank-1 wins | mean F1 |
|--:|---|--:|--:|--:|--:|
| 1 | GLADEBooleanizer ⭐ | 8 | 40% | 2 | 0.7377 |
| 2 | GLADEEncoder | 7 | 35% | 2 | 0.7359 |
| 3 | AQB | 6 | 30% | 1 | 0.7267 |
| 4 | OnlineGeneralizedBinarizer | 6 | 30% | 2 | 0.7313 |
| 5 | KnownMethodsBinarizer | 6 | 30% | 0 | 0.7147 |
| 6 | StandardBinarizerNative | 6 | 30% | 1 | 0.7163 |
| 7 | SAQT | 5 | 25% | 1 | 0.7095 |
| 8 | ResonantGradientBinarizerV2 | 5 | 25% | 1 | 0.7191 |
| 9 | AdaptiveMomentumBinarizer | 4 | 20% | 1 | 0.7067 |
| 10 | OQSB | 4 | 20% | 0 | 0.7164 |
