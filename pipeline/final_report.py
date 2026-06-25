#!/usr/bin/env python3
"""
Unified TM-vs-ML report across ALL dataset groups (cyber, original TS, UCR archive).
Reads every tm_results_*.json and joins each dataset with its meta.json (best ML F1,
kind, #classes) found under tsetlin/data or uea/data.
"""
import json, glob, os, statistics as st
ROOTS=["/workspace/ml_diagnostic/tsetlin/data", "/workspace/ml_diagnostic/uea/data"]
TMJSON=glob.glob("/workspace/ml_diagnostic/tsetlin/tm_results_*.json")

def find_dir(ds):
    for r in ROOTS:
        if os.path.isdir(f"{r}/{ds}"): return f"{r}/{ds}"
    return None

tm={}
for p in TMJSON:
    try:
        for r in json.load(open(p)):
            tm[r["dataset"]]=r          # later files win (fine; datasets are disjoint)
    except Exception: pass

rows=[]
for ds,r in tm.items():
    d=find_dir(ds)
    if not d or not os.path.isfile(f"{d}/meta.json"): continue
    meta=json.load(open(f"{d}/meta.json"))
    ml=json.load(open(f"{d}/ml.json")) if os.path.isfile(f"{d}/ml.json") else {}
    mlbest=max(((k,v["f1_macro"]) for k,v in ml.items() if "f1_macro" in v), key=lambda x:x[1], default=("?",meta.get("best_ml_f1",-1)))
    rows.append(dict(ds=ds, kind=meta.get("kind"), C=meta.get("n_classes"),
                     tm=r["tm_f1"], tm_bool=r["tm_best_booleanizer"], ml=mlbest[1], ml_model=mlbest[0]))

print(f"# Tsetlin Machine + ALL booleanizers vs ML — {len(rows)} datasets\n")
print("Per dataset: every booleanizer in the library binarizes the same preprocessed features; "
      "the best feeds a per-dataset-configured DeterministicTM. Compared to best of "
      "XGBoost/LightGBM/RandomForest/ExtraTrees on the identical split. Metric = macro-F1.\n")

tmw=sum(1 for r in rows if r["tm"]>=r["ml"])
mtm=st.mean(r["tm"] for r in rows); mml=st.mean(r["ml"] for r in rows)
print(f"- **TM ≥ ML on {tmw}/{len(rows)} datasets** ({100*tmw/len(rows):.0f}%).")
print(f"- Mean macro-F1: **TM {mtm:.4f}** vs **ML {mml:.4f}** (mean Δ {mtm-mml:+.4f}).")
# booleanizer wins
bw=collections_counter=[r["tm_bool"] for r in rows]
from collections import Counter
wc=Counter(bw)
print(f"- Booleanizers that won at least once ({len(wc)} distinct): "
      + ", ".join(f"{b}×{n}" for b,n in wc.most_common(12)) + ".\n")

# per-booleanizer mean F1 (across datasets where each ran)
allb={}
for ds,r in tm.items():
    for b,f in r.get("per_booleanizer",{}).items():
        if f is not None and f>=0: allb.setdefault(b,[]).append(f)
print("## Per-booleanizer mean TM macro-F1 (across datasets) — top 15\n")
print("| Booleanizer | mean F1 | #datasets | #wins |")
print("|---|--:|--:|--:|")
for b,vals in sorted(allb.items(), key=lambda kv:-st.mean(kv[1]))[:15]:
    print(f"| {b} | {st.mean(vals):.4f} | {len(vals)} | {wc.get(b,0)} |")

# grouped summaries
def group(name, pred):
    g=[r for r in rows if pred(r)]
    if not g: return
    w=sum(1 for r in g if r["tm"]>=r["ml"])
    print(f"\n## {name}: {len(g)} datasets — TM≥ML {w}/{len(g)}, "
          f"mean TM {st.mean(r['tm'] for r in g):.3f} vs ML {st.mean(r['ml'] for r in g):.3f}")
group("A. UCR time-series archive", lambda r: r["kind"]=="time_series" and find_dir(r["ds"]).startswith(ROOTS[1]))
group("B. Original 12 time-series", lambda r: r["kind"]=="time_series" and find_dir(r["ds"]).startswith(ROOTS[0]))
group("C. Cyber / IDS multi-class", lambda r: r["kind"]=="tabular")

# full table (sorted by TM F1)
print("\n## Full results (sorted by TM macro-F1)\n")
print("| Dataset | kind | C | TM best booleanizer | TM F1 | ML best | ML F1 | Δ | winner |")
print("|---|---|--:|---|--:|---|--:|--:|---|")
for r in sorted(rows, key=lambda r:-r["tm"]):
    d=r["tm"]-r["ml"]
    print(f"| {r['ds']} | {r['kind'][:4]} | {r['C']} | {r['tm_bool']} | {r['tm']:.3f} | "
          f"{r['ml_model']} | {r['ml']:.3f} | {d:+.3f} | {'TM' if d>=0 else 'ML'} |")
