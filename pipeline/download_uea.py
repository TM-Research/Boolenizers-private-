#!/usr/bin/env python3
"""
Download ALL 128 UCR/UEA univariate classification datasets via aeon and cache
each as /tmp/ucr_all/<name>.npz with keys X (n, length) and y (n,).

Unequal-length series are right-padded to the max length with their last value
(Weka-style); >512-length series are decimated to <=512 points to bound the
downstream booleanizer/TM cost (recorded in the npz as `decimated`).
Robust: per-dataset try/except + skip-if-cached; parallel with a small pool.
"""
import os, sys, json, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
from aeon.datasets import load_classification
from aeon.datasets.tsc_datasets import univariate

OUT = "/tmp/ucr_all"; os.makedirs(OUT, exist_ok=True)
MAXLEN = 512

def to_2d(X):
    # equal-length: ndarray (n, channels, L) -> (n, L) (univariate => channel 0)
    if isinstance(X, np.ndarray) and X.ndim == 3:
        return X[:, 0, :].astype(np.float64)
    if isinstance(X, np.ndarray) and X.ndim == 2:
        return X.astype(np.float64)
    # unequal-length: list/object array of (channels, Li) -> pad to max L
    series = [np.asarray(s)[0] if np.asarray(s).ndim == 2 else np.asarray(s).ravel() for s in X]
    L = max(len(s) for s in series)
    M = np.zeros((len(series), L))
    for i, s in enumerate(series):
        M[i, :len(s)] = s
        if len(s) < L: M[i, len(s):] = s[-1] if len(s) else 0.0
    return M

def decimate(X, maxlen):
    L = X.shape[1]
    if L <= maxlen: return X, False
    idx = np.linspace(0, L - 1, maxlen).round().astype(int)
    return X[:, idx], True

def one(name):
    out = f"{OUT}/{name}.npz"
    if os.path.exists(out):
        try:
            d = np.load(out); return (name, "cached", d["X"].shape, len(np.unique(d["y"])))
        except Exception: pass
    try:
        last = None
        for attempt in range(3):
            try:
                X, y = load_classification(name, load_equal_length=False, load_no_missing=False)
                break
            except Exception as e:
                last = e; time.sleep(2 * (attempt + 1))
        else:
            raise last
        X = to_2d(X); y = np.asarray(y)
        # drop non-finite
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X, dec = decimate(X, MAXLEN)
        np.savez_compressed(out, X=X.astype(np.float32), y=y, decimated=np.array([dec]))
        return (name, "ok", X.shape, len(np.unique(y)))
    except Exception as e:
        return (name, f"ERR:{type(e).__name__}:{str(e)[:50]}", None, None)

def main():
    names = sorted(univariate)
    print(f"Downloading {len(names)} UCR univariate datasets -> {OUT}")
    from concurrent.futures import ThreadPoolExecutor, as_completed
    status = {}
    with ThreadPoolExecutor(max_workers=4) as ex:
        for fut in as_completed([ex.submit(one, n) for n in names]):
            name, st, shp, nc = fut.result(); status[name] = st
            print(f"  {name:28s} {st if isinstance(st,str) and st.startswith('ERR') else st:8s} {shp} C={nc}", flush=True)
    json.dump(status, open(f"{OUT}/_status.json", "w"), indent=2)
    ok = sum(1 for v in status.values() if v in ("ok", "cached"))
    print(f"\n{ok}/{len(names)} datasets cached. errors: {[k for k,v in status.items() if str(v).startswith('ERR')]}")

if __name__ == "__main__":
    main()
