# Boolenizers-private-

Booleanizers (binary encoders) for **Tsetlin Machines**, benchmarked head-to-head
against gradient-boosted ML across **148 datasets** (UCR time-series archive + cyber/IDS
multi-class). Every booleanizer binarizes the *same* preprocessed split that the ML models
see; the best feeds a **per-dataset-configured DeterministicTM**.

**📄 Per-dataset results:** [results/datasets/INDEX.md](results/datasets/INDEX.md) — separate page for every dataset (top-5 booleanizers + ML).

**📊 Live results (GitHub Pages):** enable Pages on `main` → `/docs`, then open the site —
a searchable, sortable table with **per-dataset detail** (top-5 booleanizers, all booleanizers,
ML models, winner) and a booleanizer leaderboard.

## Headline results

- **Tsetlin Machine ≥ ML on 101 / 148 datasets (68 %)**
- Mean macro-F1: **TM 0.7895** vs **ML 0.7840**
- **48 booleanizers** tested; **no single one dominates** — the best is dataset-specific.
  Most consistently strong (top-5 frequency): **SAQT**, **GLADE**, **OnlineGeneralized**,
  **StandardBinarizerNative**, **ACFB**.

| Group | datasets | TM ≥ ML | mean TM | mean ML |
|---|--:|--:|--:|--:|
| UCR time-series archive | 116 | 84 | 0.777 | 0.768 |
| Original 12 time-series | 12 | 10 | 0.942 | 0.930 |
| Cyber / IDS multi-class | 20 | 7 | 0.772 | 0.788 |

## Layout

```
booleanizers/python/   44 library booleanizers (fit/transform; ThermometerEncoder API)
booleanizers/julia/    AdaptiveBooleanizers.jl — SAQT (recommended) + MWAB + OnlineSAQT
pipeline/              end-to-end: download → prep → dump bits → TM sweep → reports
results/               TM_vs_ML_REPORT.md, per-dataset top-5, raw tm_results_*.json
docs/                  GitHub Pages site (index.html + data.json)
```

## Reproduce

```bash
# 1. datasets (UCR archive via aeon + cyber caches)
python3 pipeline/download_uea.py
python3 pipeline/uea_prep.py          # split, z-norm, dump ALL booleanizer bits, ML baselines

# 2. Tsetlin sweep (Julia 1.12+, in the TsetlinMachines project)
JULIA_NUM_THREADS=128 julia --project=. --threads=128 pipeline/tm_run.jl all

# 3. reports + site data
python3 pipeline/final_report.py > results/TM_vs_ML_REPORT.md
python3 pipeline/build_site.py        # -> docs/data.json
```

## Method & caveats

- **Per-dataset config (not shared):** `clauses = clamp(2·⌈(50·C + 0.3·width)/2⌉, 256, 2000)`,
  `T = clauses/16`, `s = 5.0`, `epochs = clamp(120000/n + 8, 12, 35)`. TM = DeterministicTM
  (`balanced_rotating_sigma_server`).
- **Preprocessing fits on train only** (median-impute, winsorize for tabular; per-series
  z-normalization for time-series) — no leakage; both TM and ML see identical features.
- **Caveats:** UCR uses a stratified **70/30 split (not the canonical UCR split)**, so scores
  read higher than published numbers (the TM-vs-ML comparison stays fair); series longer than
  **512 points are decimated**. Re-run with canonical splits / full resolution for
  literature-comparable absolute numbers.
