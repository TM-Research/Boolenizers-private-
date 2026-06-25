#!/usr/bin/env python3
"""
Step-by-step ML validation & diagnostic pipeline for XGBoost + alternatives.

Four verified stages per dataset, in order:
  1. INGEST      - confirm the model receives the correct raw data
  2. PREPROCESS  - tailored transforms, verified against the raw input
  3. TRAIN       - XGBoost + LightGBM + RandomForest + ExtraTrees, per-dataset
                   hyperparameters, convergence + "is it actually learning?" check
  4. PREDICT     - accuracy / precision / recall / F1 / confusion matrix on test

CRITICAL: configuration is tailored per dataset (size, #classes, #features,
imbalance, time-series vs tabular) - never one global config.

Datasets:
  * 12 UCR real time-series  (/tmp/ucr_cache/*.npz, keys X,y)        <- run first
  * 20 cyber/IDS multiclass  (/tmp/encoders_eval_cache/*.npz, Xtr/Xte/ytr/yte)

Outputs: artifacts/results.json (everything), logs/<dataset>.log (per-dataset trace).
"""
import os, sys, json, time, glob, warnings, traceback
warnings.filterwarnings("ignore")
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[v] = "8"
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)
import xgboost as xgb
import lightgbm as lgb

UCR   = "/tmp/ucr_cache"
CYBER = "/tmp/encoders_eval_cache"
HERE  = "/workspace/ml_diagnostic"
SEED  = 42

# ------------------------------------------------------------------ registry
def list_datasets():
    ds = []
    for p in sorted(glob.glob(f"{UCR}/*.npz")):
        ds.append({"name": os.path.basename(p)[:-4], "path": p, "kind": "time_series"})
    for p in sorted(glob.glob(f"{CYBER}/*.npz")):
        ds.append({"name": os.path.basename(p)[:-4], "path": p, "kind": "tabular"})
    return ds

def load_raw(d):
    z = np.load(d["path"], allow_pickle=True)
    if d["kind"] == "time_series":            # X,y -> stratified split (TS classification)
        X, y = np.asarray(z["X"], np.float64), np.asarray(z["y"])
        # ensure each class has >=2 samples for a stratified split
        import collections
        c = collections.Counter(y)
        if min(c.values()) < 2:
            keep = np.array([yy in {k for k,v in c.items() if v>=2} for yy in y])
            X, y = X[keep], y[keep]
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=SEED, stratify=y)
    else:
        Xtr = np.asarray(z["Xtr"], np.float64); Xte = np.asarray(z["Xte"], np.float64)
        ytr = np.asarray(z["ytr"]); yte = np.asarray(z["yte"])
    return Xtr, Xte, ytr, yte

# ------------------------------------------------------------------ stage 1: INGEST
def stage_ingest(d, Xtr, Xte, ytr, yte, log):
    classes = np.unique(np.concatenate([ytr, yte]))
    cnt = {int(c): int((ytr == c).sum()) for c in classes}
    imb = (max(cnt.values()) / max(1, min(cnt.values()))) if cnt else float("inf")
    nan = int(np.isnan(Xtr).sum() + np.isnan(Xte).sum())
    inf = int(np.isinf(Xtr).sum() + np.isinf(Xte).sum())
    nconst = int(np.sum(np.nanstd(Xtr, axis=0) == 0))
    rec = dict(kind=d["kind"], n_train=int(len(ytr)), n_test=int(len(yte)),
               n_features=int(Xtr.shape[1]), n_classes=int(len(classes)),
               class_dist_train=cnt, imbalance_ratio=round(float(imb), 1),
               nan_cells=nan, inf_cells=inf, constant_features=nconst,
               dtype=str(Xtr.dtype),
               sample_row=[round(float(v), 4) for v in Xtr[0, :8]])
    flags = []
    status = "OK"
    if len(classes) < 2: flags.append("FAIL:<2 classes"); status = "FAIL"
    if len(ytr) < 20:    flags.append("FAIL:tiny train"); status = "FAIL"
    if nan or inf:       flags.append(f"WARN:{nan} nan/{inf} inf cells"); status = max(status, "WARN", key=_rank)
    if imb > 50:         flags.append(f"WARN:severe imbalance {imb:.0f}x"); status = max(status, "WARN", key=_rank)
    if min(cnt.values()) < 5: flags.append("WARN:min class <5"); status = max(status, "WARN", key=_rank)
    if nconst > 0:       flags.append(f"WARN:{nconst} constant feats"); status = max(status, "WARN", key=_rank)
    rec["flags"] = flags; rec["status"] = status
    log(f"[INGEST] {d['name']}: {d['kind']} shape=({len(ytr)}+{len(yte)})x{Xtr.shape[1]} "
        f"C={len(classes)} imb={imb:.1f}x nan={nan} inf={inf} const={nconst} -> {status}")
    log(f"         class_dist(train)={cnt}")
    log(f"         sample_row[:8]={rec['sample_row']}")
    return rec

_rank_order = {"OK":0, "WARN":1, "FAIL":2}
def _rank(s): return _rank_order.get(s, 0)

# ------------------------------------------------------------------ stage 2: PREPROCESS (tailored)
def stage_preprocess(d, Xtr, Xte, ytr, yte, ingest, log):
    steps = []
    # label encode 0..C-1 (contiguous; required by XGBoost)
    classes = np.unique(np.concatenate([ytr, yte])); remap = {c:i for i,c in enumerate(classes)}
    ytr = np.array([remap[v] for v in ytr]); yte = np.array([remap[v] for v in yte])
    steps.append("label-encode 0..C-1")

    raw_mean = float(np.nanmean(Xtr)); raw_nan = int(np.isnan(Xtr).sum()+np.isinf(Xtr).sum())

    if d["kind"] == "time_series":
        # per-series z-normalization (standard for UCR TS classification);
        # makes each series zero-mean/unit-var so trees compare shape, not offset.
        def znorm(A):
            mu = A.mean(axis=1, keepdims=True); sd = A.std(axis=1, keepdims=True); sd[sd==0]=1.0
            return (A - mu) / sd
        Xtr, Xte = znorm(Xtr), znorm(Xte)
        steps.append("per-series z-normalization (mean/std along time)")
    else:
        # tabular: inf->nan, then median impute (fit on TRAIN only), then winsorize
        # heavy-tailed columns (robust to the byte-count style outliers in IDS data).
        Xtr[np.isinf(Xtr)] = np.nan; Xte[np.isinf(Xte)] = np.nan
        med = np.nanmedian(Xtr, axis=0); med = np.where(np.isfinite(med), med, 0.0)
        ti = np.where(np.isnan(Xtr)); Xtr[ti] = np.take(med, ti[1])
        ei = np.where(np.isnan(Xte)); Xte[ei] = np.take(med, ei[1])
        steps.append(f"median-impute (train fit), filled {len(ti[0])+len(ei[0])} cells")
        # winsorize only features whose |skew|>3 (tailored, not blanket)
        from scipy.stats import skew
        sk = np.abs(skew(Xtr, axis=0, nan_policy="omit"))
        heavy = np.where(sk > 3)[0]
        if len(heavy):
            lo = np.percentile(Xtr[:, heavy], 0.5, axis=0); hi = np.percentile(Xtr[:, heavy], 99.5, axis=0)
            Xtr[:, heavy] = np.clip(Xtr[:, heavy], lo, hi); Xte[:, heavy] = np.clip(Xte[:, heavy], lo, hi)
            steps.append(f"winsorize {len(heavy)} heavy-tailed feats (|skew|>3) to [0.5,99.5]pct")
    # verification
    post_nan = int(np.isnan(Xtr).sum()+np.isinf(Xtr).sum()+np.isnan(Xte).sum()+np.isinf(Xte).sum())
    ok = (post_nan == 0) and (Xtr.shape[1] == Xte.shape[1])
    rec = dict(steps=steps, raw_nan_inf=raw_nan, post_nan_inf=post_nan,
               raw_mean=round(raw_mean,4), post_mean=round(float(Xtr.mean()),4),
               shape_preserved=bool(Xtr.shape[1]==ingest["n_features"]),
               status="OK" if ok else "FAIL")
    log(f"[PREP]   {d['name']}: steps={steps}")
    log(f"         verify: nan/inf {raw_nan}->{post_nan}  mean {raw_mean:.3f}->{Xtr.mean():.3f}  "
        f"width {ingest['n_features']}->{Xtr.shape[1]}  -> {rec['status']}")
    return rec, Xtr, Xte, ytr, yte

# ------------------------------------------------------------------ per-dataset MODEL CONFIGS (tailored)
def model_configs(n, d_feat, C, imb, kind):
    """Return {model_name: estimator}. Hyperparameters depend on dataset shape."""
    multiclass = C > 2
    # depth scales with feature count; learning-rate & trees scale (inversely) with n
    depth = 4 if d_feat <= 20 else (6 if d_feat <= 100 else 8)
    lr    = 0.30 if n < 20_000 else (0.15 if n < 100_000 else 0.08)
    ntree = 600 if n < 20_000 else (400 if n < 100_000 else 300)
    subs  = 1.0 if n < 5_000 else 0.8
    leaves= min(255, 2 ** depth - 1)
    spw   = float(np.clip(imb, 1, 100)) if (C == 2 and imb > 3) else 1.0   # imbalance handling (binary)
    cw    = "balanced" if (imb > 10) else None                            # for RF/ET
    cfg = dict(depth=depth, lr=lr, ntree=ntree, subsample=subs, leaves=leaves,
               scale_pos_weight=spw, class_weight=cw)
    models = {
        "XGBoost": xgb.XGBClassifier(
            n_estimators=ntree, max_depth=depth, learning_rate=lr, subsample=subs,
            colsample_bytree=0.8 if d_feat > 30 else 1.0, tree_method="hist",
            n_jobs=8, verbosity=0, eval_metric="mlogloss" if multiclass else "logloss",
            scale_pos_weight=spw, random_state=SEED,
            early_stopping_rounds=30),
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=ntree, max_depth=depth, num_leaves=leaves, learning_rate=lr,
            subsample=subs, colsample_bytree=0.8 if d_feat > 30 else 1.0,
            class_weight=cw, n_jobs=8, verbose=-1, random_state=SEED),
        "RandomForest": RandomForestClassifier(
            n_estimators=min(400, max(120, ntree//2)), max_depth=None,
            class_weight=cw, n_jobs=8, random_state=SEED),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=min(400, max(120, ntree//2)), max_depth=None,
            class_weight=cw, n_jobs=8, random_state=SEED),
    }
    return models, cfg

# ------------------------------------------------------------------ stage 3+4: TRAIN + PREDICT
def stage_train_predict(d, Xtr, Xte, ytr, yte, ingest, log):
    n, dim, C = len(ytr), Xtr.shape[1], ingest["n_classes"]
    imb = ingest["imbalance_ratio"]
    models, cfg = model_configs(n, dim, C, imb, d["kind"])
    log(f"[CONFIG] {d['name']}: depth={cfg['depth']} lr={cfg['lr']} ntree={cfg['ntree']} "
        f"subsample={cfg['subsample']} leaves={cfg['leaves']} scale_pos_weight={cfg['scale_pos_weight']} "
        f"class_weight={cfg['class_weight']}")
    # majority-class baseline (to verify learning)
    base = float(np.bincount(yte).max() / len(yte))
    # internal validation split for early stopping (XGB/LGBM)
    from sklearn.model_selection import train_test_split as tts
    strat = ytr if min(np.bincount(ytr)) >= 2 else None
    Xt, Xv, yt, yv = tts(Xtr, ytr, test_size=0.12, random_state=SEED, stratify=strat)

    out = {}
    for name, clf in models.items():
        r = {}
        try:
            t0 = time.perf_counter()
            if name == "XGBoost":
                clf.fit(Xt, yt, eval_set=[(Xv, yv)], verbose=False)
                r["best_iteration"] = int(getattr(clf, "best_iteration", cfg["ntree"]) or cfg["ntree"])
            elif name == "LightGBM":
                clf.fit(Xt, yt, eval_set=[(Xv, yv)],
                        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)])
                r["best_iteration"] = int(getattr(clf, "best_iteration_", cfg["ntree"]) or cfg["ntree"])
            else:
                clf.fit(Xtr, ytr)
                r["best_iteration"] = int(getattr(clf, "n_estimators", 0))
            fit_s = time.perf_counter() - t0
            tr_acc = float(accuracy_score(ytr, clf.predict(Xtr)))
            yp = clf.predict(Xte)
            te_acc = float(accuracy_score(yte, yp))
            r.update(
                fit_s=round(fit_s, 2),
                train_acc=round(tr_acc, 4),
                test_acc=round(te_acc, 4),
                test_f1_macro=round(float(f1_score(yte, yp, average="macro", zero_division=0)), 4),
                test_f1_weighted=round(float(f1_score(yte, yp, average="weighted", zero_division=0)), 4),
                test_precision_macro=round(float(precision_score(yte, yp, average="macro", zero_division=0)), 4),
                test_recall_macro=round(float(recall_score(yte, yp, average="macro", zero_division=0)), 4),
                majority_baseline=round(base, 4),
                learning_verified=bool(tr_acc > base + 0.02),
                overfit_gap=round(tr_acc - te_acc, 4),
                status="OK",
            )
            # confusion matrix (kept for low/medium C; summarized for large C)
            cm = confusion_matrix(yte, yp)
            r["confusion_matrix"] = cm.tolist() if C <= 20 else None
            r["per_class_f1_min"] = round(float(np.min(f1_score(yte, yp, average=None, zero_division=0))), 4)
            log(f"[TRAIN]  {d['name']:28s} {name:13s} fit={fit_s:6.2f}s iters={r['best_iteration']:4d} "
                f"trainAcc={tr_acc:.3f} testAcc={te_acc:.3f} F1={r['test_f1_macro']:.3f} "
                f"base={base:.3f} learn={r['learning_verified']} gap={r['overfit_gap']:+.3f}")
        except Exception as e:
            r = {"status": "FAIL", "error": f"{type(e).__name__}: {e}"}
            log(f"[TRAIN]  {d['name']} {name} FAILED: {r['error']}")
        out[name] = r
    return out

# ------------------------------------------------------------------ driver
def run_one(d):
    logs = []
    log = lambda s: logs.append(s)
    rec = {"name": d["name"], "kind": d["kind"]}
    try:
        Xtr, Xte, ytr, yte = load_raw(d)
        rec["ingest"] = stage_ingest(d, Xtr, Xte, ytr, yte, log)
        if rec["ingest"]["status"] == "FAIL":
            rec["status"] = "FAIL@ingest"
        else:
            rec["preprocess"], Xtr, Xte, ytr, yte = stage_preprocess(d, Xtr, Xte, ytr, yte, rec["ingest"], log)
            rec["models"] = stage_train_predict(d, Xtr, Xte, ytr, yte, rec["ingest"], log)
            rec["status"] = "OK"
    except Exception as e:
        rec["status"] = "ERROR"; rec["error"] = f"{type(e).__name__}: {e}"
        log(f"[ERROR] {d['name']}: {rec['error']}\n{traceback.format_exc()}")
    open(f"{HERE}/logs/{d['name']}.log", "w").write("\n".join(logs))
    return rec

def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    ds = list_datasets()
    if which == "ts":     ds = [d for d in ds if d["kind"] == "time_series"]
    elif which == "cyber":ds = [d for d in ds if d["kind"] == "tabular"]
    print(f"Running diagnostic pipeline on {len(ds)} datasets ({which})\n")
    from concurrent.futures import ProcessPoolExecutor, as_completed
    results = {}
    with ProcessPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(run_one, d): d for d in ds}
        for fut in as_completed(futs):
            r = fut.result(); results[r["name"]] = r
            if "models" in r:
                best = max(r["models"].items(), key=lambda kv: kv[1].get("test_f1_macro", -1))
                print(f"  {r['name']:30s} [{r['kind'][:4]}] {r['status']:6s} "
                      f"best={best[0]}(F1={best[1].get('test_f1_macro')})")
            else:
                print(f"  {r['name']:30s} [{r['kind'][:4]}] {r['status']}")
    out = f"{HERE}/artifacts/results_{which}.json"
    json.dump(results, open(out, "w"), indent=2)
    print(f"\nWrote {out}  ({len(results)} datasets)")

if __name__ == "__main__":
    main()
