# Evidence-Backed Design Specification — Streaming IDS Booleanizer for Tsetlin Machines

*Deep-research synthesis → engineering decisions. Every choice is justified from primary
sources and from the on-repo empirical study (148 datasets). This SUPERSEDES the initial
ad-hoc draft; it also flags where the literature contradicted that draft.*

## Decision summary

| Component | Decision | Why (evidence) |
|---|---|---|
| **Threshold placement** | **Streaming-quantile thermometer (P²)** as the accuracy-optimal; **EWMA-mean + EWMA-MAD z-levels** as the ultra-light fallback | Equal-frequency (quantile) bins carry ~equal information/bit and avoid empty/dead bins on skewed data (audio-KWS TM, arXiv:2101.11336). P² tracks quantiles in **O(1) memory/time, no stored data** (Jain & Chlamtac 1985, CACM). **Equal-width is rejected** (empty bins=dead literals on skew). **mean±kσ is rejected**: network traffic is self-similar with *infinite-variance* ON/OFF sources (Willinger 1997; Leland 1994) → variance collapses/explodes. MAD has a 50% breakdown point (Croux & Dehon). |
| **Encoding** | **Thermometer/cumulative**, K≈8 magnitude bits/feature, low-cardinality→unique-value thresholds, +1 `is-zero` bit for zero-inflated | Thermometer lets one literal = a half-line and AND of two = an interval, matching TM clause grammar; one-hot forces OR-of-bins a conjunction can't express (Abeyrathna 2019, arXiv:1905.04199; Granmo 2018). Even 1–2 quantile bits are highly discriminative → budget conservatively (arXiv:2101.11336). |
| **Drift detection** | **Page-Hinkley per feature on the hot path** (O(1)); on trigger, **boost the quantile-tracker adaptation** (decoupled re-fit). ADWIN/ECDD as principled alternatives | Page-Hinkley is the cheapest O(1) CUSUM trigger (River). ADWIN gives provable bounds + a fresh window to re-fit (Bifet & Gavaldà 2007). Architecture from Gama et al. survey (ACM CSUR 2014): freeze thresholds while quiescent, re-fit only on a trigger → keep adaptation off the per-sample hot path. |
| **Temporal literals** | **3 compact bits/feature**: `trend` (x>slow-EWMA), `burst` (\|Δx\|>scale), `drift` (PH active) | Traffic is temporally local/bursty (Leland 1994), so order-aware recency bits add signal. **Caveat from our own 148-dataset study:** temporal-indicator bits *hurt* on shuffled tables — they only pay off on genuinely ordered/drifting streams, which is the target deployment. Keep them few (RQ4: no dimensionality blow-up). |
| **Class imbalance** | Quantile cut-points (not central mean) + **macro-F1** objective; tails get their own bins | Central mean/median thresholds sit in the benign mass and bury rare attacks (Electronics 14(1):69). Quantile placement keeps tail structure; macro-F1 stops accuracy from hiding missed attacks. |
| **Evaluation** | **Prequential test-then-train** on ordered streams; **inject covariate drift**; metrics = macro-F1, accuracy, P, R, **throughput (samples/s), latency (µs), state memory (bytes), #literals, post-drift adaptation/recovery time, TM train+inference time** | Standard data-stream evaluation (Gama; Bifet *ML for Data Streams*). Stationary-vs-drift gap isolates the value of online adaptation. |

## What the research CHANGED vs the initial draft

The initial draft (ASIB-R) placed thermometer bits at **fixed normal-quantile z-levels**
`Φ⁻¹(k/(K+1))` of the robust z-score — which silently assumes approximate normality
*after* robust scaling. The literature is explicit that IDS features are **heavy-tailed,
multimodal, infinite-variance** (Willinger 1997), so fixed-z bins are **not** equal-frequency
and waste bits in the tails. The evidence therefore favors a **distribution-free streaming
quantile (P²)** thermometer for the magnitude code. We keep the robust-MAD variant as the
*ultra-lightweight* option and **benchmark both** — directly answering RQ3 (can cheap stats
replace quantiles?) with data rather than assertion.

## The two specified variants (to be compared)

- **ASIB-Q (accuracy-optimal):** per feature, a **P² streaming-quantile** tracker (K markers,
  O(1)) → K-bit thermometer at empirical deciles; Page-Hinkley drift → on trigger, widen P²
  step / partial-reset markers; + `is-zero` + 3 temporal bits. Distribution-free.
- **ASIB-R (ultra-light):** per feature, **EWMA mean + EWMA-MAD** → K-bit thermometer at fixed
  normal z-levels; Page-Hinkley → α-boost; + temporal bits. Cheapest (8 scalars/feature),
  approximate.

## One-paragraph buildable spec

For each feature maintain O(1) state: a P² quantile tracker with K=8 markers (ASIB-Q) **or**
an EWMA mean + EWMA-MAD pair (ASIB-R), a previous-value register, and a Page-Hinkley
accumulator. Per streamed sample, emit a K-bit **thermometer** against the current
quantile/robust thresholds (distribution-free for ASIB-Q), plus `is-zero`, `trend`
(x>slow-mean), and `burst` (|Δx|>scale) bits, and a `drift` bit set while Page-Hinkley is
firing; low-cardinality features use unique-value thresholds instead. Update the tracker,
the EWMA(s), and Page-Hinkley in constant time; on a Page-Hinkley trigger, temporarily boost
the tracker's adaptation rate so thresholds snap to the new regime, then decay back. Encode
labels for a Tsetlin Machine and evaluate **prequentially** on the ordered stream under both
stationary and injected-drift conditions, reporting macro-F1 alongside throughput, latency,
memory, #literals, and post-drift recovery time.

## Top 3 design risks

1. **Temporal/drift bits can hurt when order is not informative** (confirmed on our 148-dataset
   tabular study). Mitigation: make temporal bits optional and report stationary-vs-drift
   separately; they are justified only for truly ordered streams.
2. **P² instability on multimodal / abruptly-shifting features** (P² is heuristic, no error
   bound; Jain & Chlamtac). Mitigation: Page-Hinkley-triggered marker reset; ADWIN fallback
   that supplies a clean re-fit window.
3. **Reconstructed "streams" from shuffled benchmark tables may not reflect real temporal
   structure**, inflating or deflating the adaptive advantage. Mitigation: use original record
   order where available and inject *controlled* drift so the adaptation claim is measured
   against a known ground truth, not an artifact.

### Primary sources
Granmo 2018 (arXiv:1804.01508); Abeyrathna et al. 2019 (arXiv:1905.04199); Jain & Chlamtac
1985 (CACM, P²); Ma, Muthukrishnan & Sandler (arXiv:1407.1121, Frugal); Greenwald & Khanna
2001 (SIGMOD); Dunning & Ertl 2019 (arXiv:1902.04023, t-digest); Bifet & Gavaldà 2007 (SDM,
ADWIN); Ross et al. 2012 (arXiv:1212.6018, ECDD); Gama et al. 2014 (ACM CSUR 46(4), drift
survey); Leland et al. 1994 (IEEE/ACM ToN); Willinger et al. 1997 (SIGCOMM); audio-KWS TM
2021 (arXiv:2101.11336); IDS concept-drift survey 2024 (Eng. Appl. AI 137).
