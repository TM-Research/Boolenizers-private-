#!/usr/bin/env python3
"""
Unified prep for ALL UCR univariate datasets (/tmp/ucr_all/*.npz):
per dataset -> stratified split, per-series z-norm, save X/y CSV, dump ALL
booleanizer bits, and compute ML baselines (XGB/LGBM/RF/ET) on the same split.
Output: uea/data/<name>/  (same layout the Julia TM runner consumes).

Parallel across datasets; robust per-encoder/per-model try/except.
Large datasets are stratified-capped (train<=8000, test<=3000) to bound TM cost.
"""
import os, sys, json, glob, time, warnings, inspect, collections
warnings.filterwarnings("ignore")
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"): os.environ[v]="2"
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import f1_score, accuracy_score
import xgboost as xgb, lightgbm as lgb
sys.path.insert(0,"/workspace")
import encoders

SRC="/tmp/ucr_all"; OUT="/workspace/ml_diagnostic/uea/data"; SEED=42
CAP_TR, CAP_TE = 8000, 3000
os.makedirs(OUT, exist_ok=True)
SKIP={"ThermometerEncoder"}
ENC_NAMES=[n for n in encoders.__all__ if n not in SKIP]

def to_u8(B):
    B=np.asarray(B); B=B.reshape(len(B),-1) if B.ndim==1 else B; return (B!=0).astype(np.uint8)
def dump(path,B):
    B=to_u8(B); n,w=B.shape
    with open(path,"wb") as f: np.array([n,w],dtype="<i4").tofile(f); B.reshape(-1).astype(np.uint8).tofile(f)
    return w
def strat_cap(X,y,n,rng):
    if n>=len(y): return X,y
    cls,cnt=np.unique(y,return_counts=True); alloc=np.maximum(1,np.round(n*cnt/cnt.sum()).astype(int))
    idx=np.concatenate([rng.choice(np.where(y==c)[0],min(k,(y==c).sum()),replace=False) for c,k in zip(cls,alloc)])
    rng.shuffle(idx); return X[idx],y[idx]
def znorm(A):
    mu=A.mean(1,keepdims=True); sd=A.std(1,keepdims=True); sd[sd==0]=1; return (A-mu)/sd

def ml_configs(n,d_feat,C,imb):
    depth=6 if d_feat<=100 else 8; lr=0.3 if n<20000 else 0.15; ntree=400 if n<20000 else 300
    cw="balanced" if imb>10 else None
    return {"XGBoost":xgb.XGBClassifier(n_estimators=ntree,max_depth=depth,learning_rate=lr,tree_method="hist",
                n_jobs=2,verbosity=0,random_state=SEED),
            "LightGBM":lgb.LGBMClassifier(n_estimators=ntree,max_depth=depth,num_leaves=min(255,2**depth-1),
                learning_rate=lr,class_weight=cw,n_jobs=2,verbose=-1,random_state=SEED),
            "RandomForest":RandomForestClassifier(n_estimators=200,class_weight=cw,n_jobs=2,random_state=SEED),
            "ExtraTrees":ExtraTreesClassifier(n_estimators=200,class_weight=cw,n_jobs=2,random_state=SEED)}

def one(path):
    name=os.path.basename(path)[:-4]; d=os.path.join(OUT,name); os.makedirs(d,exist_ok=True)
    if os.path.isfile(f"{d}/ml.json") and os.path.isfile(f"{d}/all_bits_status.json"):
        return (name,"cached")
    try:
        rng=np.random.default_rng(SEED)
        z=np.load(path,allow_pickle=True); X=np.asarray(z["X"],np.float64); y=np.asarray(z["y"])
        c=collections.Counter(y); keep=np.array([yy in {k for k,v in c.items() if v>=2} for yy in y])
        X,y=X[keep],y[keep]
        if len(np.unique(y))<2 or len(y)<20: return (name,"too-small")
        Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.3,random_state=SEED,stratify=y)
        Xtr,ytr=strat_cap(Xtr,ytr,CAP_TR,rng); Xte,yte=strat_cap(Xte,yte,CAP_TE,rng)
        cls=np.unique(np.concatenate([ytr,yte])); rm={cc:i for i,cc in enumerate(cls)}
        ytr=np.array([rm[v] for v in ytr]); yte=np.array([rm[v] for v in yte])
        Xtr,Xte=znorm(Xtr),znorm(Xte)
        np.savetxt(f"{d}/X_train.csv",Xtr,delimiter=",",fmt="%.6g"); np.savetxt(f"{d}/X_test.csv",Xte,delimiter=",",fmt="%.6g")
        np.savetxt(f"{d}/y_train.csv",ytr,fmt="%d"); np.savetxt(f"{d}/y_test.csv",yte,fmt="%d")
        # all booleanizer bits
        status={}
        for nm in ENC_NAMES:
            cls_=getattr(encoders,nm,None)
            if not (inspect.isclass(cls_) and hasattr(cls_,"fit") and hasattr(cls_,"transform")): continue
            if os.path.isfile(f"{d}/bits_{nm}_train.bin"): status[nm]="cached"; continue
            try:
                e=cls_()
                try: e.fit(Xtr)
                except TypeError: e.fit(Xtr,ytr)
                Btr=to_u8(e.transform(Xtr)); Bte=to_u8(e.transform(Xte))
                w=min(Btr.shape[1],Bte.shape[1])
                if w==0: raise ValueError("zero-width")
                dump(f"{d}/bits_{nm}_train.bin",Btr[:,:w]); dump(f"{d}/bits_{nm}_test.bin",Bte[:,:w]); status[nm]=int(w)
            except Exception as ex: status[nm]=f"ERR:{type(ex).__name__}"
        json.dump(status,open(f"{d}/all_bits_status.json","w"),indent=2)
        # ML baselines
        n,df,C=len(ytr),Xtr.shape[1],len(cls); imb=float(np.bincount(ytr).max()/max(1,np.bincount(ytr).min()))
        ml={}
        for mn,clf in ml_configs(n,df,C,imb).items():
            try: clf.fit(Xtr,ytr); yp=clf.predict(Xte); ml[mn]={"f1_macro":round(float(f1_score(yte,yp,average="macro",zero_division=0)),4),"acc":round(float(accuracy_score(yte,yp)),4)}
            except Exception as ex: ml[mn]={"error":str(ex)[:60]}
        best=max((m["f1_macro"] for m in ml.values() if "f1_macro" in m),default=-1)
        json.dump(ml,open(f"{d}/ml.json","w"),indent=2)
        json.dump({"name":name,"kind":"time_series","n_train":n,"n_test":len(yte),"n_features":df,
                   "n_classes":C,"imbalance":round(imb,1),"best_ml_f1":best},open(f"{d}/meta.json","w"),indent=2)
        ok=sum(1 for v in status.values() if isinstance(v,int) or v=="cached")
        return (name,f"ok bits={ok}/{len(status)} ML_best={best:.3f}")
    except Exception as ex:
        return (name,f"ERR:{type(ex).__name__}:{str(ex)[:50]}")

def main():
    paths=sorted(glob.glob(f"{SRC}/*.npz")); paths=[p for p in paths if not os.path.basename(p).startswith("_")]
    print(f"UEA prep over {len(paths)} datasets")
    from concurrent.futures import ProcessPoolExecutor, as_completed
    done=0
    with ProcessPoolExecutor(max_workers=10) as ex:
        for fut in as_completed([ex.submit(one,p) for p in paths]):
            name,st=fut.result(); done+=1
            print(f"  [{done:3d}/{len(paths)}] {name:30s} {st}",flush=True)
    print("done.")

if __name__=="__main__": main()
