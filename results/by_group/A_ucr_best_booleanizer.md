# Best booleanizer for the UCR time-series archive (116 datasets)

Aggregated by taking the **top-5 booleanizers for each dataset**.

## By top-5 frequency (how often a method is among the 5 best)

| Rank | Booleanizer | in top-5 | % of 116 | rank-1 wins | mean F1 |
|--:|---|--:|--:|--:|--:|
| 1 | **ACFB** | **34** | **29%** | **14** | 0.7362 |
| 2 | DecisionTreeBinarizer | 25 | 22% | 3 | 0.7349 |
| 3 | NTEUniform | 22 | 19% | 8 | 0.7364 |
| 4 | AdaptiveQuantileBinarizer | 20 | 17% | 4 | 0.7339 |
| 5 | DriftRobustBinarizer | 19 | 16% | 3 | 0.7337 |
| 5 | GLADEBooleanizer | 19 | 16% | 3 | 0.7297 |
| 5 | StandardBinarizerNative | 19 | 16% | 3 | 0.7343 |
| 5 | OnlineGeneralizedBinarizer | 19 | 16% | 3 | 0.7356 |
| 5 | SpectralStabilityBinarizer | 19 | 16% | 0 | 0.7364 |
| 10 | SAQT | 18 | 16% | 3 | 0.7318 |

## Verdict

**ACFB (AutoCorrelation Feature Binarizer) is the best booleanizer for UCR time-series** —
most frequent in the top-5 (29%, ~1.5× the runner-up) and most rank-1 wins (14).
NTEUniform and GLADEEncoder are next on outright wins (8 each).

**Caveat:** mean macro-F1 is statistically flat across the field (0.732–0.737), because
many UCR datasets saturate and booleanizers tie. "Best" = *most often among the strongest*,
where ACFB leads clearly — not meaningfully higher on average. Recommended order:
**ACFB → DecisionTreeBinarizer → NTEUniform → AdaptiveQuantile.**
