#!/usr/bin/env python3
"""
Streaming IDS booleanizer benchmark (prequential-style, stationary + injected drift).

For each IDS dataset x {stationary, drift} x booleanizer:
  * encode train+test as an ordered stream (online encoders adapt; batch encoders
    fit on the train prefix then freeze),
  * measure booleanizer THROUGHPUT (samples/s), LATENCY (us/sample), STATE MEMORY
    (bytes), #LITERALS (width),
  * train a real Tsetlin Machine (tmu) on the train bits, evaluate on test bits:
    macro-F1, accuracy, precision, recall, TM train time, TM inference time.

The stationary-vs-drift macro-F1 GAP isolates online-adaptation value: static
encoders degrade under drift; adaptive ones (ASIB) should not.

Compared booleanizers (the objective's list, Python-available):
  ASIB-Q, ASIB-R (new) | AdaptiveGaussian, OnlineGeneralized (online) |
  AdaptiveQuantile(AQB), GLADE, ResonantGradientV2, StandardBinarizerNative (batch).
"""
import os, sys, json, time, warnings, glob, tracemalloc
warnings.filterwarnings("ignore")
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"): os.environ[v]="8"
import numpy as np
sys.path.insert(0, "/workspace")
from encoders.asib import ASIB_Q, ASIB_R
from encoders.asib_v2 import ASIBv2
from encoders.asib_v3 import ASIBAdaptive
from encoders import (AdaptiveGaussian, OnlineGeneralizedBinarizer,
                      AdaptiveQuantileBinarizer, GLADEBooleanizer,
                      ResonantGradientBinarizerV2, StandardBinarizerNative,
                      OnlineQuantileSignalBinarizer, AdaptiveMomentumBinarizer)
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import xgboost as xgb, lightgbm as lgb
from tmu.models.classification.vanilla_classifier import TMClassifier

CACHE="/tmp/encoders_eval_cache"; OUT="/workspace/ml_diagnostic/streaming_ids"; SEED=42
DATASETS=["anf-iot","cic-iov-2024-multiclass","nids-bench-2026",
          "edge-iiotset-multiclass","cicmaldroid-2020-multiclass"]
N_TRAIN, N_TEST = 7000, 4000
CLAUSES, T_, S_, EPOCHS = 400, 25, 5.0, 8
rng=np.random.default_rng(SEED)

# booleanizer registry: name -> (factory, kind)  kind in {"online","batch"}
BOOL = {
  "ASIB-v3":           (lambda: ASIBAdaptive(),                    "online"),  # proposed: fully self-parameterizing
  "ASIB-v2":           (lambda: ASIBv2(warmup=600, kmax=12),       "online"),  # proposed (iteration 2)
  "ASIB-R":            (lambda: ASIB_R(K=8),                       "online"),  # proposed (light, robust)
  "AdaptiveGaussian":  (lambda: AdaptiveGaussian(K=8),             "online"),
  "OnlineGeneralized": (lambda: OnlineGeneralizedBinarizer(K_q=8), "online"),
  "OQSB":              (lambda: OnlineQuantileSignalBinarizer(K_q=8, K_s=4), "online"),
  "AdaptiveMomentum":  (lambda: AdaptiveMomentumBinarizer(K=8),    "batch"),
  "AQB":               (lambda: AdaptiveQuantileBinarizer(max_bits_per_feature=8), "batch"),
  "GLADE":             (lambda: GLADEBooleanizer(n_bins=8),        "batch"),
  "RGB2":              (lambda: ResonantGradientBinarizerV2(max_bits_per_feature=8), "batch"),
  "StandardNative":    (lambda: StandardBinarizerNative(K=8),      "batch"),
}

def run_ml(Xtr, ytr, Xte, yte, C):
    """ML baselines on raw standardized features (per scenario)."""
    sc=StandardScaler().fit(Xtr); Xtr2,Xte2=sc.transform(Xtr),sc.transform(Xte)
    cw="balanced"
    models={"ML:XGBoost":xgb.XGBClassifier(n_estimators=300,max_depth=8,learning_rate=0.15,
              tree_method="hist",n_jobs=8,verbosity=0,random_state=SEED),
            "ML:LightGBM":lgb.LGBMClassifier(n_estimators=300,max_depth=8,num_leaves=127,
              learning_rate=0.15,class_weight=cw,n_jobs=8,verbose=-1,random_state=SEED),
            "ML:ExtraTrees":ExtraTreesClassifier(n_estimators=300,class_weight=cw,n_jobs=8,random_state=SEED),
            "ML:RandomForest":RandomForestClassifier(n_estimators=300,class_weight=cw,n_jobs=8,random_state=SEED)}
    rows=[]
    for nm,clf in models.items():
        t0=time.perf_counter(); clf.fit(Xtr2,ytr); tr=time.perf_counter()-t0
        t0=time.perf_counter(); yp=clf.predict(Xte2); it=time.perf_counter()-t0
        rows.append(dict(booleanizer=nm,kind="ml",width=Xtr.shape[1],
            macro_f1=round(float(f1_score(yte,yp,average="macro",zero_division=0)),4),
            accuracy=round(float(accuracy_score(yte,yp)),4),
            precision=round(float(precision_score(yte,yp,average="macro",zero_division=0)),4),
            recall=round(float(recall_score(yte,yp,average="macro",zero_division=0)),4),
            tm_train_s=round(tr,2), tm_infer_s=round(it,3)))
    return rows

def load(name):
    z=np.load(f"{CACHE}/{name}.npz", allow_pickle=True)
    X=np.vstack([z["Xtr"], z["Xte"]]).astype(np.float64)
    y=np.concatenate([z["ytr"], z["yte"]]).astype(int)
    # cap, preserve order (simulate packet arrival)
    m=min(len(y), N_TRAIN+N_TEST); X,y=X[:m],y[:m]
    X[~np.isfinite(X)]=0.0
    cl=np.unique(y); rm={c:i for i,c in enumerate(cl)}; y=np.array([rm[v] for v in y])
    return X, y

def inject_drift(X, strength=2.5):
    """Gradual covariate drift over the stream (scale grows + mean shifts)."""
    n=len(X); t=np.linspace(0,1,n)[:,None]; sd=X.std(0,keepdims=True)+1e-9
    return X*(1.0+strength*t) + strength*t*sd

def to_u8(B):
    B=np.asarray(B); B=B.reshape(len(B),-1) if B.ndim==1 else B; return (B!=0).astype(np.uint8)

def state_bytes(enc, width):
    if hasattr(enc,"state_bytes"):
        try: return int(enc.state_bytes())
        except Exception: pass
    # batch encoders: stored thresholds (≈ width floats)
    for a in ("thresholds_","_thresh","feature_thresholds_","unique_values"):
        v=getattr(enc,a,None)
        if v is not None:
            try: return int(np.asarray(v, dtype=object).size*8) if a=="feature_thresholds_" else int(np.asarray(v).nbytes)
            except Exception: pass
    return int(width*8)

def encode(name, factory, kind, Xtr, Xte):
    enc=factory()
    streaming = hasattr(enc, "update_transform_row")        # ASIB: pure stream, no fit
    if kind=="batch" or not streaming:
        enc.fit(Xtr)                                        # batch fit, or init online encoder
    Btr=to_u8(enc.transform(Xtr))
    t0=time.perf_counter(); Bte=to_u8(enc.transform(Xte)); dt=time.perf_counter()-t0
    w=min(Btr.shape[1],Bte.shape[1]); Btr,Bte=Btr[:,:w],Bte[:,:w]
    thr=N_TEST/dt if dt>0 else 0; lat=1e6*dt/len(Xte)
    return Btr, Bte, dict(width=int(w), throughput=round(thr,0), latency_us=round(lat,2),
                          state_bytes=state_bytes(enc,w))

def run_tm(Btr, ytr, Bte, yte, classes):
    Xtr=np.ascontiguousarray(Btr.astype(np.uint32)); Xte=np.ascontiguousarray(Bte.astype(np.uint32))
    Ytr=ytr.astype(np.uint32)
    m=TMClassifier(number_of_clauses=CLAUSES, T=T_, s=S_, platform='CPU', weighted_clauses=True)
    t0=time.perf_counter(); m.fit(Xtr, Ytr, epochs=EPOCHS); tr_t=time.perf_counter()-t0
    t0=time.perf_counter(); yp=m.predict(Xte); inf_t=time.perf_counter()-t0
    return dict(macro_f1=round(float(f1_score(yte,yp,average="macro",zero_division=0)),4),
                accuracy=round(float(accuracy_score(yte,yp)),4),
                precision=round(float(precision_score(yte,yp,average="macro",zero_division=0)),4),
                recall=round(float(recall_score(yte,yp,average="macro",zero_division=0)),4),
                tm_train_s=round(tr_t,2), tm_infer_s=round(inf_t,3))

def main():
    which=sys.argv[1] if len(sys.argv)>1 else "all"
    ds_list=[d for d in DATASETS if which=="all" or d==which]
    results=[]
    for ds in ds_list:
        X,y=load(ds); classes=np.unique(y)
        for scen in ["stationary","drift"]:
            Xs = X if scen=="stationary" else inject_drift(X)
            Xtr,Xte=Xs[:N_TRAIN], Xs[N_TRAIN:]; ytr,yte=y[:N_TRAIN], y[N_TRAIN:]
            print(f"\n=== {ds} [{scen}]  train {Xtr.shape} test {Xte.shape} C={len(classes)} ===", flush=True)
            for name,(factory,kind) in BOOL.items():
                try:
                    Btr,Bte,bm=encode(name,factory,kind,Xtr,Xte)
                    tm=run_tm(Btr,ytr,Bte,yte,classes)
                    row=dict(dataset=ds,scenario=scen,booleanizer=name,kind=kind,**bm,**tm)
                    results.append(row)
                    print(f"  {name:18s}[{kind[:3]}] F1={tm['macro_f1']:.3f} acc={tm['accuracy']:.3f} "
                          f"| w={bm['width']:4d} thr={bm['throughput']:>8.0f}/s lat={bm['latency_us']:6.1f}us "
                          f"mem={bm['state_bytes']:>6d}B | TMtrain={tm['tm_train_s']}s", flush=True)
                except Exception as e:
                    print(f"  {name}: ERROR {type(e).__name__}: {e}", flush=True)
                    results.append(dict(dataset=ds,scenario=scen,booleanizer=name,error=str(e)[:80]))
            # ML baselines (raw features)
            try:
                for r in run_ml(Xtr,ytr,Xte,yte,len(classes)):
                    r.update(dataset=ds,scenario=scen); results.append(r)
                    print(f"  {r['booleanizer']:18s}[ml ] F1={r['macro_f1']:.3f} acc={r['accuracy']:.3f} "
                          f"| train={r['tm_train_s']}s", flush=True)
            except Exception as e:
                print(f"  ML ERROR: {e}", flush=True)
    os.makedirs(OUT,exist_ok=True)
    json.dump(results, open(f"{OUT}/stream_results.json","w"), indent=1)
    print(f"\nWrote {OUT}/stream_results.json ({len(results)} rows)")

if __name__=="__main__": main()
