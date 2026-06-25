# =============================================================================
# Tsetlin Machine vs ML — run multiple booleanizers per dataset through a
# per-dataset-configured DeterministicTM, on the SAME splits the ML models used.
#
# Booleanizers: SAQT, MWAB (native Julia) + AQB, OQSB, NTEUniform (Python bits).
# TM config is tailored per dataset (clauses/T/epochs from #classes, width, n) —
# never a shared config. Reports macro-F1 per booleanizer + the best, and carries
# the best-ML F1 (from tm_prep) for the head-to-head.
#
#   JULIA_NUM_THREADS=128 julia --project=. --threads=128 \
#       /workspace/ml_diagnostic/tsetlin/tm_run.jl [ts|cyber|all]
# =============================================================================
using DelimitedFiles, Statistics, Printf
using Statistics: mean

const TMROOT = "/workspace/Tsetlin_TM-main 2"
include(joinpath(TMROOT, "src", "booleanizers", "AdaptiveBooleanizers.jl"))
using .AdaptiveBooleanizers
push!(LOAD_PATH, TMROOT)
import Pkg; Pkg.activate(TMROOT; io=devnull)
using TsetlinMachines
include(joinpath(TMROOT, "src", "models", "DeterministicTM.jl"))
const D = DeterministicTM

const DATA = get(ENV, "DATADIR", "/workspace/ml_diagnostic/tsetlin/data")
const OUTTAG = get(ENV, "OUTTAG", "")   # appended to output filename
const SEED = 1
# auto-discover every dumped booleanizer (bits_<name>_train.bin) in a dataset dir
discover_bools(dir) = sort(unique([m.captures[1] for m in
    (match(r"^bits_(.+)_train\.bin$", f) for f in readdir(dir)) if m !== nothing]))

# per-dataset adaptive TM config (NOT shared) — from dataset shape
function tm_config(n, width, C)
    clauses = clamp(2*cld(round(Int, 50*C + 0.3*width), 2), 256, 2000)
    T = clamp(round(Int, clauses/16), 8, 120)
    epochs = clamp(round(Int, 120_000 / max(n,1)) + 8, 12, 35)
    return clauses, T, 5.0, epochs
end

macro_f1(yt, yp, cs) = mean(begin
    tp=sum((yp.==c).&(yt.==c)); fp=sum((yp.==c).&(yt.!=c)); fn=sum((yp.!=c).&(yt.==c))
    p=tp+fp==0 ? 0.0 : tp/(tp+fp); r=tp+fn==0 ? 0.0 : tp/(tp+fn)
    p+r==0 ? 0.0 : 2p*r/(p+r)
end for c in cs)

read_bits(path) = open(path) do io
    n=Int(read(io,Int32)); w=Int(read(io,Int32)); raw=read(io,n*w)
    rows=Vector{Vector{Bool}}(undef,n)
    @inbounds for i in 1:n; off=(i-1)*w; b=Vector{Bool}(undef,w)
        for j in 1:w; b[j]=raw[off+j]!=0; end; rows[i]=b; end
    rows,w
end

function run_det(tr, te, ytr, yte, cs, cfg)
    Xv=D.TMInput[D.TMInput(b) for b in tr]; Xt=D.TMInput[D.TMInput(b) for b in te]
    cl,T,s,ep=cfg
    m=D.TMClassifier(Xv[1], ytr, cl, T, s; max_included_literals=32,
                     feedback_mode=:balanced_rotating_sigma_server, seed=SEED)
    for _ in 1:ep; D.train_hogwild!(m, Xv, ytr; non_targets=:all); end
    macro_f1(yte, D.predict(m, Xt; index=false), cs)
end

read_json_num(path, key) = begin   # tiny extractor for best_ml_f1 from meta.json
    s = read(path, String); m = match(Regex("\"$key\"\\s*:\\s*([0-9.eE+-]+)"), s)
    m === nothing ? -1.0 : parse(Float64, m.captures[1])
end

which = length(ARGS) >= 1 ? ARGS[1] : "all"
kind_of(dir) = occursin("\"time_series\"", read(joinpath(dir,"meta.json"), String)) ? "time_series" : "tabular"
all_ds = sort([d for d in readdir(DATA) if isdir(joinpath(DATA,d)) && isfile(joinpath(DATA,d,"meta.json"))])
if which == "ts";    all_ds = [d for d in all_ds if kind_of(joinpath(DATA,d))=="time_series"]
elseif which=="cyber"; all_ds = [d for d in all_ds if kind_of(joinpath(DATA,d))=="tabular"]; end
println("Threads=", Threads.nthreads(), "  datasets=", length(all_ds), " (", which, ")")
results = Any[]
for ds in all_ds
    dir=joinpath(DATA,ds); isfile(joinpath(dir,"X_train.csv")) || continue
    Xtr=readdlm(joinpath(dir,"X_train.csv"),',',Float64); Xte=readdlm(joinpath(dir,"X_test.csv"),',',Float64)
    ytr=[parse(Int,x) for x in readlines(joinpath(dir,"y_train.csv"))]
    yte=[parse(Int,x) for x in readlines(joinpath(dir,"y_test.csv"))]
    cs=sort(unique(vcat(ytr,yte)))
    best_ml = read_json_num(joinpath(dir,"meta.json"), "best_ml_f1")

    enc=Dict{String,Tuple{Vector{Vector{Bool}},Vector{Vector{Bool}}}}()
    try; sa=fit_saqt(Xtr); enc["SAQT"]=(transform_saqt(sa,Xtr),transform_saqt(sa,Xte)); catch; end
    try; mw=fit_mwab(Xtr); enc["MWAB"]=(transform_mwab_stream(mw,Xtr),transform_mwab_stream(mw,Xte;warmup=Xtr)); catch; end
    for en in discover_bools(dir)               # ALL dumped booleanizers
        tp=joinpath(dir,"bits_$(en)_train.bin"); ep=joinpath(dir,"bits_$(en)_test.bin")
        (isfile(tp)&&isfile(ep)) || continue
        try; enc[en]=(read_bits(tp)[1], read_bits(ep)[1]); catch; end
    end

    f1s=Dict{String,Float64}()
    for (en,(tr,te)) in enc
        cfg=tm_config(length(ytr), length(tr[1]), length(cs))
        f1s[en] = try; run_det(tr,te,ytr,yte,cs,cfg); catch e; -1.0; end
    end
    bestb = isempty(f1s) ? ("none",-1.0) : reduce((a,b)->a[2]>=b[2] ? a : b, collect(f1s))
    @printf("%-34s C=%2d  TM_best=%-10s %.3f | bestML %.3f | %s\n",
            ds, length(cs), bestb[1], bestb[2], best_ml,
            bestb[2]>=best_ml ? "TM≥ML" : "ML>TM")
    push!(results, Dict("dataset"=>ds, "n_classes"=>length(cs), "tm_f1"=>round(bestb[2],digits=4),
          "tm_best_booleanizer"=>bestb[1], "ml_f1"=>best_ml,
          "per_booleanizer"=>Dict(k=>round(v,digits=4) for (k,v) in f1s)))
    flush(stdout)
end

# write json
function jval(v)
    v isa String && return "\"$v\""
    v isa Dict && return "{"*join(["\"$k\": $(jval(x))" for (k,x) in v],", ")*"}"
    return string(v)
end
open("/workspace/ml_diagnostic/tsetlin/tm_results_$(which)$(OUTTAG).json","w") do f
    print(f,"[\n")
    for (i,r) in enumerate(results); print(f,"  ",jval(r), i==length(results) ? "\n" : ",\n"); end
    print(f,"]\n")
end
println("\nWrote tm_results.json (", length(results), " datasets)")
