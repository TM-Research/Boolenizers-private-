# Best booleanizer for MULTICLASS datasets (C ≥ 3) — by macro-F1

106 multiclass datasets (UCR archive 83 + original TS 3 + cyber/IDS 20).
Metric is **macro-F1** (not accuracy), aggregated from the per-dataset top-5.

| Booleanizer | in top-5 | % | rank-1 wins | mean F1 |
|---|--:|--:|--:|--:|
| **ACFB** | 27 | 25% | **12** | 0.6994 |
| DecisionTreeBinarizer | 24 | 23% | 5 | 0.6961 |
| StandardBinarizerNative | 22 | 21% | 4 | 0.7069 |
| **GLADEEncoder** | 21 | 20% | 9 | **0.7118** |
| OnlineGeneralizedBinarizer | 21 | 20% | 4 | 0.7106 |
| NTEUniform | 20 | 19% | 7 | 0.6958 |
| GLADEBooleanizer | 20 | 19% | 4 | 0.7072 |
| SAQT | 19 | 18% | 2 | 0.7058 |

## Verdict
- **ACFB** wins the most datasets outright (12 rank-1, top-5 on 25%) → best single default.
- **GLADEEncoder** has the highest mean macro-F1 (0.712 vs ACFB 0.699) and 2nd-most wins (9)
  → best on average.

**Recommendation:** try **ACFB** first; use **GLADEEncoder** when you want the best average
performance. (DecisionTreeBinarizer and StandardBinarizerNative are consistent backups.)

*Note:* 20 cyber datasets recorded some encoders under short aliases (`AQB`, `OQSB`) while UCR
used full class names — same encoders, counted separately — so AdaptiveQuantile's true total
is marginally higher than its row shows. The ACFB/GLADE leaders are unaffected.
