# Boolenizers-private-

Booleanizers (binary encoders) for **Tsetlin Machines**, benchmarked head-to-head
against gradient-boosted ML across **148 datasets** (UCR time-series archive + cyber/IDS
multi-class). Every booleanizer binarizes the *same* preprocessed split that the ML models
see; the best feeds a **per-dataset-configured DeterministicTM**.

## 🔗 Links

- **🌐 Live site (GitHub Pages):** https://tm-research.github.io/Boolenizers-private-/ — searchable/sortable table, **group filter**, click any dataset for its **top-5 booleanizers + ML** detail page.
- **📄 Per-dataset pages (Markdown):** [results/datasets/INDEX.md](results/datasets/INDEX.md) — a separate file per dataset (top-5 booleanizers + ML results).
- **📊 Results by group (kept separate):**
  - [A · UCR time-series archive (116)](results/by_group/A_ucr_archive.md)
  - [B · Original 12 time-series](results/by_group/B_original_ts.md)
  - [C · Cyber / IDS multi-class (20)](results/by_group/C_cyber_ids.md)
- **🏆 Best booleanizer by group:** [results/by_group/BEST_BOOLEANIZER_BY_GROUP.md](results/by_group/BEST_BOOLEANIZER_BY_GROUP.md) — A·UCR (ACFB), B·Original-TS (GLADEEncoder), C·Cyber-IDS (GLADEBooleanizer)
- **📈 Combined report:** [results/TM_vs_ML_REPORT.md](results/TM_vs_ML_REPORT.md) · **booleanizer findings:** [results/SAQT_FINDINGS.md](results/SAQT_FINDINGS.md)
- **🧩 Booleanizer source:** [Python (44)](booleanizers/python/) · [Julia SAQT/MWAB](booleanizers/julia/AdaptiveBooleanizers.jl) · **pipeline:** [pipeline/](pipeline/)

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
