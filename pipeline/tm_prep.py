#!/usr/bin/env python3
"""
Prep for the Tsetlin-Machine vs ML comparison — SAME datasets/splits/preprocessing
as the ML diagnostic, so TM-F1 and ML-F1 are head-to-head on identical data.

Per dataset:
  * load same cache (UCR /tmp/ucr_cache X,y  | cyber /tmp/encoders_eval_cache Xtr..),
  * stratified cap (cyber 30k/10k; UCR full) — TM is slower than trees,
  * tailored preprocessing (TS: per-series z-norm; tabular: median-impute + winsorize),
  * save processed X/y CSV (for the Julia booleanizers),
  * dump booleanizer bit matrices (AQB, OQSB, NTEUniform) for the Julia TM,
  * recompute ML F1 (XGBoost/LightGBM/RandomForest/ExtraTrees) on this exact split.

Writes tsetlin/data/<name>/{X_*.csv,y_*.csv,bits_*_*.bin,ml.json,meta.json}.
"""
import os, sys, json, glob, time, warnings, collections
warnings.filterwarnings("ignore")
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"): os.environ[v]="8"
import numpy as np
from scipy.stats import skew
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import f1_score, accuracy_score
import xgboost as xgb, lightgbm as lgb
sys.path.insert(0, "/workspace")
from encoders import (AdaptiveQuantileBinarizer, OnlineQuantileSignalBinarizer, NTEUniform)

UCR="/tmp/ucr_cache"; CYBER="/tmp/encoders_eval_cache"
OUT="/workspace/ml_diagnostic/tsetlin/data"; SEED=42
CAP_TR, CAP_TE = 30000, 10000
os.makedirs(OUT, exist_ok=True)
rng=np.random.default_rng(SEED)

def datasets():
    ds=[{"name":os.path.basename(p)[:-4],"path":p,"kind":"time_series"} for p in sorted(glob.glob(f"{UCR}/*.npz"))]
    ds+=[{"name":os.path.basename(p)[:-4],"path":p,"kind":"tabular"} for p in sorted(glob.glob(f"{CYBER}/*.npz"))]
    return ds

def strat_cap(X,y,n):
    if n>=len(y): return X,y
    cls,cnt=np.unique(y,return_counts=True)
    alloc=np.maximum(1,np.round(n*cnt/cnt.sum()).astype(int))
    idx=np.concatenate([rng.choice(np.where(y==c)[0],min(k,(y==c).sum()),replace=False) for c,k in zip(cls,alloc)])
    rng.shuffle(idx); return X[idx],y[idx]

def load(d):
    z=np.load(d["path"],allow_pickle=True)
    if d["kind"]=="time_series":
        X,y=np.asarray(z["X"],np.float64),np.asarray(z["y"])
        c=collections.Counter(y); keep=np.array([yy in {k for k,v in c.items() if v>=2} for yy in y])
        X,y=X[keep],y[keep]
        Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.3,random_state=SEED,stratify=y)
    else:
        Xtr,Xte=np.asarray(z["Xtr"],np.float64),np.asarray(z["Xte"],np.float64)
        ytr,yte=np.asarray(z["ytr"]),np.asarray(z["yte"])
        Xtr,ytr=strat_cap(Xtr,ytr,CAP_TR); Xte,yte=strat_cap(Xte,yte,CAP_TE)
    return Xtr,Xte,ytr,yte

def preprocess(d,Xtr,Xte):
    if d["kind"]=="time_series":
        def zn(A):
            mu=A.mean(1,keepdims=True); sd=A.std(1,keepdims=True); sd[sd==0]=1; return (A-mu)/sd
        return zn(Xtr),zn(Xte)
    Xtr=Xtr.copy(); Xte=Xte.copy()
    Xtr[np.isinf(Xtr)]=np.nan; Xte[np.isinf(Xte)]=np.nan
    med=np.nanmedian(Xtr,axis=0); med=np.where(np.isfinite(med),med,0.)
    ti=np.where(np.isnan(Xtr)); Xtr[ti]=np.take(med,ti[1]); ei=np.where(np.isnan(Xte)); Xte[ei]=np.take(med,ei[1])
    sk=np.abs(skew(Xtr,axis=0,nan_policy="omit")); heavy=np.where(sk>3)[0]
    if len(heavy):
        lo=np.percentile(Xtr[:,heavy],0.5,axis=0); hi=np.percentile(Xtr[:,heavy],99.5,axis=0)
        Xtr[:,heavy]=np.clip(Xtr[:,heavy],lo,hi); Xte[:,heavy]=np.clip(Xte[:,heavy],lo,hi)
    return Xtr,Xte

ENC={"AQB":lambda:AdaptiveQuantileBinarizer(max_bits_per_feature=10),
     "OQSB":lambda:OnlineQuantileSignalBinarizer(K_q=8,K_s=4),
     "NTEUniform":lambda:NTEUniform()}
def to_u8(B):
    B=np.asarray(B); B=B.reshape(len(B),-1) if B.ndim==1 else B; return (B!=0).astype(np.uint8)
def dump(path,B):
    B=to_u8(B); n,w=B.shape
    with open(path,"wb") as f: np.array([n,w],dtype="<i4").tofile(f); B.reshape(-1).astype(np.uint8).tofile(f)
    return w

def ml_configs(n,d_feat,C,imb):
    depth=4 if d_feat<=20 else (6 if d_feat<=100 else 8); lr=0.3 if n<20000 else 0.15
    ntree=400 if n<20000 else 300; cw="balanced" if imb>10 else None
    spw=float(np.clip(imb,1,100)) if (C==2 and imb>3) else 1.0
    return {"XGBoost":xgb.XGBClassifier(n_estimators=ntree,max_depth=depth,learning_rate=lr,tree_method="hist",
                n_jobs=8,verbosity=0,scale_pos_weight=spw,random_state=SEED),
            "LightGBM":lgb.LGBMClassifier(n_estimators=ntree,max_depth=depth,num_leaves=min(255,2**depth-1),
                learning_rate=lr,class_weight=cw,n_jobs=8,verbose=-1,random_state=SEED),
            "RandomForest":RandomForestClassifier(n_estimators=200,class_weight=cw,n_jobs=8,random_state=SEED),
            "ExtraTrees":ExtraTreesClassifier(n_estimators=200,class_weight=cw,n_jobs=8,random_state=SEED)}

def main():
    which=sys.argv[1] if len(sys.argv)>1 else "all"
    ds=datasets()
    if which=="ts": ds=[d for d in ds if d["kind"]=="time_series"]
    elif which=="cyber": ds=[d for d in ds if d["kind"]=="tabular"]
    print(f"TM-prep over {len(ds)} datasets ({which})")
    for d in ds:
        t0=time.time(); name=d["name"]; dd=os.path.join(OUT,name); os.makedirs(dd,exist_ok=True)
        Xtr,Xte,ytr,yte=load(d)
        classes=np.unique(np.concatenate([ytr,yte])); rm={c:i for i,c in enumerate(classes)}
        ytr=np.array([rm[v] for v in ytr]); yte=np.array([rm[v] for v in yte])
        Xtr,Xte=preprocess(d,Xtr,Xte)
        np.savetxt(f"{dd}/X_train.csv",Xtr,delimiter=",",fmt="%.6g"); np.savetxt(f"{dd}/X_test.csv",Xte,delimiter=",",fmt="%.6g")
        np.savetxt(f"{dd}/y_train.csv",ytr,fmt="%d"); np.savetxt(f"{dd}/y_test.csv",yte,fmt="%d")
        # booleanizer bits
        widths={}
        for en,mk in ENC.items():
            try:
                e=mk(); e.fit(Xtr); Btr=to_u8(e.transform(Xtr)); Bte=to_u8(e.transform(Xte))
                w=min(Btr.shape[1],Bte.shape[1]); Btr,Bte=Btr[:,:w],Bte[:,:w]
                dump(f"{dd}/bits_{en}_train.bin",Btr); dump(f"{dd}/bits_{en}_test.bin",Bte); widths[en]=int(w)
            except Exception as ex: widths[en]=f"ERR:{type(ex).__name__}"
        # ML on SAME split
        n,df,C=len(ytr),Xtr.shape[1],len(classes)
        imb=float(np.bincount(ytr).max()/max(1,np.bincount(ytr).min()))
        ml={}
        for mn,clf in ml_configs(n,df,C,imb).items():
            try:
                clf.fit(Xtr,ytr); yp=clf.predict(Xte)
                ml[mn]={"f1_macro":round(float(f1_score(yte,yp,average="macro",zero_division=0)),4),
                        "acc":round(float(accuracy_score(yte,yp)),4)}
            except Exception as ex: ml[mn]={"error":str(ex)}
        best_ml=max((m for m in ml.values() if "f1_macro" in m),key=lambda x:x["f1_macro"],default={"f1_macro":-1})
        json.dump(ml,open(f"{dd}/ml.json","w"),indent=2)
        json.dump({"name":name,"kind":d["kind"],"n_train":n,"n_test":len(yte),"n_features":df,
                   "n_classes":C,"imbalance":round(imb,1),"old_widths":widths,
                   "best_ml_f1":best_ml["f1_macro"]},open(f"{dd}/meta.json","w"),indent=2)
        print(f"  {name:34s} [{d['kind'][:4]}] {n}x{df} C={C} imb={imb:.0f} "
              f"bestML_F1={best_ml['f1_macro']:.3f} bits={widths} ({time.time()-t0:.0f}s)")
    print("done.")

if __name__=="__main__": main()
