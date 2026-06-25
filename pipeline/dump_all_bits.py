#!/usr/bin/env python3
"""
Dump bits for EVERY encoder in encoders.__all__ on each prepared dataset, reusing
the already-saved preprocessed splits (data/<name>/X_*.csv). Parallel across
datasets; robust per-encoder try/except; skips encoders already dumped.
"""
import os, sys, glob, json, time, warnings, inspect
warnings.filterwarnings("ignore")
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"): os.environ[v]="2"
import numpy as np
sys.path.insert(0,"/workspace")
import encoders

DATA="/workspace/ml_diagnostic/tsetlin/data"
SKIP={"ThermometerEncoder"}
NAMES=[n for n in encoders.__all__ if n not in SKIP]

def to_u8(B):
    B=np.asarray(B); B=B.reshape(len(B),-1) if B.ndim==1 else B; return (B!=0).astype(np.uint8)
def dump(path,B):
    B=to_u8(B); n,w=B.shape
    with open(path,"wb") as f: np.array([n,w],dtype="<i4").tofile(f); B.reshape(-1).astype(np.uint8).tofile(f)
    return w

def do_dataset(ds):
    d=os.path.join(DATA,ds)
    if not os.path.isfile(f"{d}/X_train.csv"): return (ds,{})
    Xtr=np.loadtxt(f"{d}/X_train.csv",delimiter=","); Xte=np.loadtxt(f"{d}/X_test.csv",delimiter=",")
    if Xtr.ndim==1: Xtr=Xtr.reshape(-1,1); Xte=Xte.reshape(-1,1)
    ytr=np.loadtxt(f"{d}/y_train.csv").astype(int)
    status={}
    for nm in NAMES:
        cls=getattr(encoders,nm,None)
        if not (inspect.isclass(cls) and hasattr(cls,"fit") and hasattr(cls,"transform")): continue
        trp=f"{d}/bits_{nm}_train.bin"
        if os.path.isfile(trp) and os.path.isfile(f"{d}/bits_{nm}_test.bin"):
            status[nm]="cached"; continue
        try:
            e=cls()
            try: e.fit(Xtr)
            except TypeError: e.fit(Xtr,ytr)
            Btr=to_u8(e.transform(Xtr)); Bte=to_u8(e.transform(Xte))
            w=min(Btr.shape[1],Bte.shape[1]); Btr,Bte=Btr[:,:w],Bte[:,:w]
            if w==0: raise ValueError("zero-width")
            dump(trp,Btr); dump(f"{d}/bits_{nm}_test.bin",Bte); status[nm]=int(w)
        except Exception as ex:
            status[nm]=f"ERR:{type(ex).__name__}"
    json.dump(status,open(f"{d}/all_bits_status.json","w"),indent=2)
    ok=sum(1 for v in status.values() if isinstance(v,int) or v=="cached")
    print(f"  {ds:34s} {ok}/{len(status)} encoders ok",flush=True)
    return (ds,status)

def main():
    which=sys.argv[1] if len(sys.argv)>1 else "all"
    dss=sorted(d for d in os.listdir(DATA) if os.path.isdir(f"{DATA}/{d}") and os.path.isfile(f"{DATA}/{d}/meta.json"))
    def kind(ds): return json.load(open(f"{DATA}/{ds}/meta.json"))["kind"]
    if which=="ts": dss=[d for d in dss if kind(d)=="time_series"]
    elif which=="cyber": dss=[d for d in dss if kind(d)=="tabular"]
    print(f"Dumping ALL {len(NAMES)} encoders over {len(dss)} datasets ({which})")
    from concurrent.futures import ProcessPoolExecutor, as_completed
    with ProcessPoolExecutor(max_workers=8) as ex:
        for fut in as_completed([ex.submit(do_dataset,ds) for ds in dss]): fut.result()
    print("done.")

if __name__=="__main__": main()
