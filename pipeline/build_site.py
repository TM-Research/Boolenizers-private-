#!/usr/bin/env python3
"""Generate data.json for the GitHub Pages site from the TM + ML results."""
import json, glob, os, statistics as st, collections
TS="/workspace/ml_diagnostic/tsetlin"; ROOTS=[f"{TS}/data","/workspace/ml_diagnostic/uea/data"]
OUT="/workspace/ml_diagnostic/site"

tm={}
for p in glob.glob(f"{TS}/tm_results_*.json"):
    for r in json.load(open(p)): tm[r["dataset"]]=r
def ddir(ds):
    for r in ROOTS:
        if os.path.isdir(f"{r}/{ds}"): return f"{r}/{ds}"
    return None

rows=[]; top5c=collections.Counter(); winc=collections.Counter(); meanf={}
for ds,r in tm.items():
    d=ddir(ds)
    if not d or not os.path.isfile(f"{d}/meta.json"): continue
    meta=json.load(open(f"{d}/meta.json"))
    ml=json.load(open(f"{d}/ml.json")) if os.path.isfile(f"{d}/ml.json") else {}
    mlm={k:v["f1_macro"] for k,v in ml.items() if "f1_macro" in v}
    mlbest=max(mlm.items(), key=lambda x:x[1], default=("?",meta.get("best_ml_f1",-1)))
    pb={k:v for k,v in r.get("per_booleanizer",{}).items() if v is not None and v>=0}
    ranked=sorted(pb.items(), key=lambda x:(-x[1], x[0]))
    top5=ranked[:5]
    for b,f in top5: top5c[b]+=1
    if ranked: winc[ranked[0][0]]+=1
    for b,f in pb.items(): meanf.setdefault(b,[]).append(f)
    tmf=ranked[0][1] if ranked else -1; tmb=ranked[0][0] if ranked else "?"
    rows.append(dict(dataset=ds, kind=meta.get("kind"), C=meta.get("n_classes"),
        n_train=meta.get("n_train"), n_test=meta.get("n_test"), n_features=meta.get("n_features"),
        imbalance=meta.get("imbalance"), tm_best=tmb, tm_f1=round(tmf,4),
        ml_best=mlbest[0], ml_f1=round(mlbest[1],4), delta=round(tmf-mlbest[1],4),
        winner=("TM" if tmf>=mlbest[1] else "ML"),
        top5=[[b,round(f,4)] for b,f in top5],
        allb=[[b,round(f,4)] for b,f in ranked],
        ml_models={k:round(v,4) for k,v in sorted(mlm.items(),key=lambda x:-x[1])}))

booleanizers=[]
for b,vals in meanf.items():
    booleanizers.append(dict(name=b, mean_f1=round(st.mean(vals),4), n=len(vals),
                             top5=top5c.get(b,0), wins=winc.get(b,0)))
booleanizers.sort(key=lambda x:-x["mean_f1"])

summary=dict(
    n_datasets=len(rows),
    tm_ge_ml=sum(1 for r in rows if r["winner"]=="TM"),
    mean_tm=round(st.mean(r["tm_f1"] for r in rows),4),
    mean_ml=round(st.mean(r["ml_f1"] for r in rows),4),
    groups={g:dict(n=sum(1 for r in rows if grp(r)==g),
                   tm_ge=sum(1 for r in rows if grp(r)==g and r["winner"]=="TM"))
            for g in ["UCR archive","Original TS","Cyber/IDS"]} if False else {},
    distinct_winners=len(winc), distinct_booleanizers=len(meanf))

def grp(r):
    d=ddir(r["dataset"])
    if r["kind"]=="tabular": return "Cyber/IDS"
    return "UCR archive" if d and d.startswith(ROOTS[1]) else "Original TS"
for g in ["UCR archive","Original TS","Cyber/IDS"]:
    gr=[r for r in rows if grp(r)==g]
    summary.setdefault("groups",{})[g]=dict(n=len(gr), tm_ge=sum(1 for r in gr if r["winner"]=="TM"),
        mean_tm=round(st.mean([r["tm_f1"] for r in gr]),4) if gr else 0,
        mean_ml=round(st.mean([r["ml_f1"] for r in gr]),4) if gr else 0)

rows.sort(key=lambda r:-r["tm_f1"])
json.dump(dict(summary=summary, datasets=rows, booleanizers=booleanizers,
               top5_common=top5c.most_common(15)),
          open(f"{OUT}/data.json","w"))
print(f"wrote data.json: {len(rows)} datasets, {len(booleanizers)} booleanizers")
print("TM>=ML:", summary["tm_ge_ml"], "/", summary["n_datasets"])
