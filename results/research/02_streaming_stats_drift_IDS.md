# Lightweight online stats + concept drift for streaming IDS binarization (synthesis)

**Streaming quantiles (cost/accuracy).** P² (Jain & Chlamtac 1985): 5 markers/quantile,
O(1) time & memory, heuristic. Frugal-1U/2U (Ma et al.): 1–2 ints/quantile, O(1) — best
edge fit. Greenwald-Khanna: ε-approx, memory grows with N. t-digest (Dunning): relative
tail error, mergeable, heavier. Exponential histograms (Datar et al.): sliding-window/
recency-aware. **Only P²/Frugal are true O(1) per-sample.**

**Robust scale, not variance.** EWMA mean is O(1). But on bursty/heavy-tailed traffic,
mean±kσ collapses/explodes (variance instability). MAD / EWMA-mean-abs-deviation has a 50%
breakdown point and stays defined even without finite variance → stable bands (Croux &
Dehon; Wikipedia MAD).

**Drift detection (cheap triggers).** Page-Hinkley (CUSUM, O(1)) — cheapest. ECDD (Ross
et al. 2012, EWMA control chart, constant FP rate). ADWIN (Bifet & Gavaldà 2007, provable
bounds, hands you the fresh window). DDM/EDDM (error-based). Survey: Gama et al., ACM CSUR
46(4) 2014. Architecture: O(1) label-free detector per feature; freeze thresholds when
quiescent, re-fit only on a trigger → keep re-adaptation off the hot path.

**IDS traffic characteristics.** Self-similar/bursty with infinite-variance ON/OFF sources
(Leland et al. 1994; Willinger et al. 1997) → mean/variance unstable, Gaussian thresholds
miscalibrated. Temporal locality → order matters (not i.i.d.). Concept drift from evolving
attacks stales static models. Severe class imbalance → central thresholds bury rare
attacks. Flow features are cheap single-pass state.

**Design principles:** (1) O(1) hot path (EWMA + EWMA-MAD + Page-Hinkley); (2) robust MAD
scale not variance; (3) quantile-like, imbalance-aware cut-points; (4) order-aware recency
weighting; (5) cheap drift trigger with decoupled re-fit; (6) per-feature bounded state.

Sources: Jain & Chlamtac (CACM); Ma et al. (arXiv:1407.1121); Greenwald-Khanna (SIGMOD);
Dunning & Ertl (arXiv:1902.04023); Welford 1962; Bifet & Gavaldà ADWIN (SDM); Ross et al.
ECDD (arXiv:1212.6018); Gama et al. (ACM CSUR 2014); Leland et al. (IEEE/ACM ToN);
Willinger et al. (SIGCOMM'95); IDS concept-drift survey (Eng. Appl. AI 137, 2024).
