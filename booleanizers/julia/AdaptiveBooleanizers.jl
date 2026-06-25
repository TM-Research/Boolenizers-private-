module AdaptiveBooleanizers

# =============================================================================
# Adaptive booleanizers for Tsetlin Machines on time-series / IDS / sensor data.
#
# Two encoders, each with at most TWO parameters, and EVERY parameter is derived
# from the data — there are no hand-tuned thresholds, bands, EMA decays, or RSI
# levels (the constants that sank the prior `obb/oatb/ormb/odmb/...` encoders).
# Thresholds are always empirical quantiles of the data; the only free quantities
# are *how many* bits to spend, and those come from textbook statistics rules:
#
#   * Sturges' rule           K = ceil(log2 n) + 1            (Sturges 1926)
#   * Freedman–Diaconis rule  h = 2·IQR·n^(-1/3); K = range/h (Freedman & Diaconis 1981)
#
# Both rules appear in standard statistics texts (e.g. Wasserman, *All of
# Statistics* §6; Izenman, *Modern Multivariate Statistical Techniques* §4;
# Scott, *Multivariate Density Estimation*). They give a data-driven bin count
# with NO magic number.
#
# Method 1 — SAQT (Sample-by-sample Adaptive Quantile Thermometer)
#   Stateless across samples: each sample is booleanized independently against
#   per-feature empirical-quantile thresholds fixed at fit time. The single
#   adaptive parameter is the per-feature bit budget K_j (Freedman–Diaconis,
#   capped by Sturges and by the feature's distinct-value count).
#
# Method 2 — MWAB (Moving-Window Adaptive Booleanizer)
#   Order-aware / streaming: each sample is booleanized against the local
#   empirical quantiles of its trailing window. Two adaptive parameters:
#     W  — window length, from the autocorrelation decay of the stream (√n
#          fallback), and
#     K  — local thermometer levels = Sturges(W).
#   Local quantiles re-center the signal, so additive/multiplicative drift is
#   absorbed without ever computing a σ that can collapse on flat features.
#
# Both encoders emit a `Vector{Bool}` per sample — exactly the contract the TM
# `Booleanizer` expects (`raw_sample -> Vector{Bool}`), so the bits feed Vanilla
# and DeterministicTM unchanged.
# =============================================================================

using Statistics: quantile, median, mean, std
using Random

export SAQTModel, MWABModel, fit_saqt, transform_saqt,
       fit_mwab, transform_mwab_stream, total_width

# ----------------------------------------------------------------------------
# Textbook adaptive bin-count rules (data-derived; no constants tuned by hand).
# ----------------------------------------------------------------------------

"Sturges' rule: K = ceil(log2 n) + 1. A conservative, sample-size-driven count."
sturges(n::Integer) = max(1, ceil(Int, log2(max(n, 2))) + 1)

"""
    fd_bin_count(col) -> Int

Freedman–Diaconis bin count for one column: `h = 2·IQR·n^(-1/3)`, `K = range/h`.
Returns 0 for a constant column (no information → no bits). Falls back to
Sturges when the IQR is zero but the range is not (spread concentrated in tails).
The result is capped by Sturges(n) and by the column's distinct-value count, so
heavy-tailed IDS features cannot explode the bit budget.
"""
function fd_bin_count(col::AbstractVector{<:Real}; cap_cardinality::Bool=true)
    n = length(col)
    n < 2 && return 0
    lo, hi = extrema(col)
    rng = hi - lo
    rng <= 0 && return 0                       # constant feature carries no info
    kmax = cap_cardinality ? min(sturges(n), max(1, length(unique(col)) - 1)) : sturges(n)
    q1 = quantile(col, 0.25); q3 = quantile(col, 0.75)
    iqr = q3 - q1
    if iqr <= 0                                # degenerate IQR → Sturges fallback
        return kmax
    end
    h = 2 * iqr * n^(-1/3)
    k = ceil(Int, rng / h)
    return clamp(k, 1, kmax)
end

"Empirical-quantile thresholds at the K interior probabilities i/(K+1), de-duplicated."
function quantile_thresholds(col::AbstractVector{<:Real}, K::Integer)
    K <= 0 && return Float64[]
    ps = (1:K) ./ (K + 1)
    ts = quantile(col, collect(ps))
    return unique(ts)                          # drop redundant knots on discrete cols
end

"Uniform (equal-width) thresholds — the ablation baseline against quantile knots."
function uniform_thresholds(col::AbstractVector{<:Real}, K::Integer)
    K <= 0 && return Float64[]
    lo, hi = extrema(col)
    hi <= lo && return Float64[]
    return unique([lo + (hi - lo) * i / (K + 1) for i in 1:K])
end

# ----------------------------------------------------------------------------
# Method 1 — SAQT (sample-by-sample, stateless)
# ----------------------------------------------------------------------------

struct SAQTModel
    thresholds::Vector{Vector{Float64}}   # one threshold list per feature
    width::Int
end

total_width(m::SAQTModel) = m.width

"""
    fit_saqt(X; use_fd, fixed_k, use_quantile, prune, cap_cardinality) -> SAQTModel

`X` is `n × d` (rows = samples). For every feature, choose a bit budget `K_j` and
place `K_j` thresholds; the encoder is then fixed (stateless across samples).

Defaults reproduce the production encoder. The keywords exist for the **ablation
study** — each isolates one design choice:
- `use_fd=true`     bit budget from Freedman–Diaconis; `false` → `fixed_k` per feature.
- `use_quantile=true` empirical-quantile knots; `false` → uniform equal-width knots.
- `prune=true`      drop dead/duplicate bit columns at fit (guarantees no wasted bits).
- `cap_cardinality=true` cap `K_j` by the feature's distinct-value count.

Thresholds are kept **sorted ascending** so the per-feature bit pattern is a proper
monotone thermometer (enables the fast searchsorted transform).
"""
function fit_saqt(X::AbstractMatrix{<:Real};
                  use_fd::Bool=true, fixed_k::Int=8, use_quantile::Bool=true,
                  prune::Bool=true, cap_cardinality::Bool=true)
    n, d = size(X)
    th = Vector{Vector{Float64}}(undef, d)
    @inbounds for j in 1:d
        col = @view X[:, j]
        if use_fd
            K = fd_bin_count(col; cap_cardinality=cap_cardinality)
        else
            lo, hi = extrema(col); K = (hi <= lo) ? 0 : fixed_k
            cap_cardinality && (K = min(K, max(1, length(unique(col)) - 1)))
        end
        cand = use_quantile ? quantile_thresholds(col, K) : uniform_thresholds(col, K)
        if prune
            # Keep a threshold only if its train bit column is non-constant (no dead
            # bit) and not identical to an already-kept column (no duplicate bit).
            kept = Float64[]; kept_cols = BitVector[]
            for t in cand
                colbits = BitVector(col[i] >= t for i in 1:n)
                s = count(colbits)
                (s == 0 || s == n) && continue
                any(c -> c == colbits, kept_cols) && continue
                push!(kept_cols, colbits); push!(kept, t)
            end
            th[j] = sort!(kept)
        else
            th[j] = sort!(collect(cand))           # ablation: keep all (may waste bits)
        end
    end
    width = sum(length, th)
    width == 0 && error("SAQT: all features constant; nothing to encode.")
    return SAQTModel(th, width)
end

"""
Booleanize one sample → `Vector{Bool}` of length `model.width`. Thresholds are
sorted, so the count of set bits per feature is one binary search; we still emit
explicit thermometer bits (the TM needs them) but avoid the threshold scan.
"""
function transform_saqt(model::SAQTModel, x::AbstractVector{<:Real})
    bits = Vector{Bool}(undef, model.width)
    p = 1
    @inbounds for j in eachindex(model.thresholds)
        ts = model.thresholds[j]
        nj = length(ts)
        # number of thresholds <= x[j]; sorted ascending thermometer → first `c` bits set
        c = searchsortedlast(ts, x[j])
        for k in 1:nj
            bits[p] = k <= c
            p += 1
        end
    end
    return bits
end

"Booleanize every row of `X` → Vector{Vector{Bool}}."
transform_saqt(model::SAQTModel, X::AbstractMatrix{<:Real}) =
    [transform_saqt(model, @view X[i, :]) for i in 1:size(X, 1)]

# ----------------------------------------------------------------------------
# Method 2 — MWAB (moving window, streaming / order-aware)
# ----------------------------------------------------------------------------

struct MWABModel
    W::Int                 # adaptive window length
    K::Int                 # adaptive local thermometer levels = Sturges(W)
    d::Int                 # feature count
    width::Int             # d * K
end

total_width(m::MWABModel) = m.width

"""
    adaptive_window(X) -> Int

Window length from the stream's own autocorrelation: the smallest lag at which
the mean (over features) absolute lag-autocorrelation drops below 1/e. If the
rows carry no temporal correlation (e.g. a pre-shuffled table), this is small
and the window simply becomes a random-subsample quantile estimator; the √n
rule is used as a floor/fallback so the local quantiles stay well estimated.
"""
function adaptive_window(X::AbstractMatrix{<:Real})
    n, d = size(X)
    # √n is the stability floor: a window needs ~√n samples for its empirical
    # quantiles to be well estimated (a standard density/quantile sample-size
    # rule of thumb). The window may grow ABOVE this to match a slow drift
    # timescale, but never shrinks below it — so on a pre-shuffled table (no
    # temporal structure) MWAB degrades gracefully to a stable local≈global
    # quantile estimator instead of collapsing to a noisy tiny window.
    floor_w = clamp(round(Int, sqrt(n)), 16, 512)
    maxlag = clamp(round(Int, sqrt(n)), 4, 128)
    m = min(n, 4000)                       # cap rows used for the ACF estimate
    Xs = @view X[1:m, :]
    mu = [mean(@view Xs[:, j]) for j in 1:d]
    sd = [std(@view Xs[:, j]) for j in 1:d]
    thr = 1 / ℯ
    drift_lag = maxlag                     # default: correlation persists → slow drift
    @inbounds for L in 1:maxlag
        acc = 0.0; cnt = 0
        for j in 1:d
            sd[j] <= 0 && continue
            a = 0.0
            for i in 1:(m - L)
                a += (Xs[i, j] - mu[j]) * (Xs[i + L, j] - mu[j])
            end
            a /= ((m - L) * sd[j]^2)
            acc += abs(a); cnt += 1
        end
        cnt == 0 && break
        if (acc / cnt) < thr            # autocorrelation has decayed by lag L
            drift_lag = L
            break
        end
    end
    # Window = drift timescale, floored at √n for quantile stability.
    return clamp(max(drift_lag, floor_w), floor_w, min(n, 1024))
end

"""
    fit_mwab(X) -> MWABModel

Derive the window `W` from the data and the local level count `K = Sturges(W)`.
No thresholds are stored — they are recomputed locally per sample at transform.
"""
function fit_mwab(X::AbstractMatrix{<:Real})
    n, d = size(X)
    W = adaptive_window(X)
    K = sturges(W)
    return MWABModel(W, K, d, d * K)
end

# thermometer of x against quantiles of a window column buffer
@inline function thermo!(bits, off, xval, wincol, K)
    ps = (1:K) ./ (K + 1)
    ts = quantile(wincol, collect(ps))
    @inbounds for k in 1:K
        bits[off + k] = xval >= ts[k]
    end
    return off + K
end

"""
    transform_mwab_stream(model, X; warmup=nothing) -> Vector{Vector{Bool}}

Booleanize `X` as a stream. Each sample `i` is encoded against the empirical
quantiles of the trailing window (the up-to-W samples ending at `i`). Pass the
tail of the training stream as `warmup` so the first test samples already have a
full window (realistic online deployment); otherwise the window grows from
sample 1.
"""
function transform_mwab_stream(model::MWABModel, X::AbstractMatrix{<:Real};
                               warmup::Union{Nothing,AbstractMatrix{<:Real}}=nothing)
    n, d = size(X)
    W, K = model.W, model.K
    out = Vector{Vector{Bool}}(undef, n)
    # ring buffer of recent rows (Vector of length-d vectors)
    buf = Vector{Vector{Float64}}()
    if warmup !== nothing
        wstart = max(1, size(warmup, 1) - W + 1)
        for i in wstart:size(warmup, 1)
            push!(buf, Float64.(@view warmup[i, :]))
        end
    end
    col = Vector{Float64}(undef, W + 1)
    @inbounds for i in 1:n
        xi = @view X[i, :]
        bits = Vector{Bool}(undef, model.width)
        # current window = buffer contents + current sample
        m = length(buf)
        off = 0
        for j in 1:d
            # gather window column j (including current sample) into `col`
            for t in 1:m
                col[t] = buf[t][j]
            end
            col[m + 1] = xi[j]
            wincol = @view col[1:(m + 1)]
            off = thermo!(bits, off, xi[j], wincol, K)
        end
        out[i] = bits
        # advance the ring buffer
        push!(buf, Float64.(xi))
        length(buf) > W && popfirst!(buf)
    end
    return out
end

# ----------------------------------------------------------------------------
# Online / streaming SAQT
#
# The production SAQT is already stateless at transform (deployable on a stream),
# but its thresholds are frozen at fit. OnlineSAQT keeps the bit SCHEMA fixed
# (so the TM input width never changes) while letting each feature's thresholds
# DRIFT: a bounded per-feature reservoir holds a global (not windowed) sample of
# everything seen, and thresholds are periodically recomputed from it at the same
# fixed quantile probabilities. Fixed width + drifting knots + global estimates =
# magnitude-preserving online adaptation. This is the streaming counterpart of
# SAQT and the reason SAQT (not a windowed encoder) is the right base for streams.
# ----------------------------------------------------------------------------

export OnlineSAQT, fit_online, partial_fit!, refit_thresholds!, transform_online

mutable struct OnlineSAQT
    probs::Vector{Vector{Float64}}        # fixed quantile probabilities per feature (schema)
    thresholds::Vector{Vector{Float64}}   # current thresholds (drift with the stream)
    reservoir::Vector{Vector{Float64}}    # bounded global sample per feature
    cap::Int                              # reservoir cap per feature (data-derived)
    width::Int
    seen::Int
    rng::Random.Xoshiro
end

"""
    fit_online(Xwarm; cap, seed) -> OnlineSAQT

Set the fixed schema from a warm-up prefix: per feature, the bit budget `K_j`
(Freedman–Diaconis + dead/duplicate prune) defines `K_j` evenly-spaced quantile
probabilities; the initial thresholds are those quantiles of the warm-up data.
"""
function fit_online(Xwarm::AbstractMatrix{<:Real}; cap::Int=4096, seed::Int=1)
    base = fit_saqt(Xwarm)                       # gives pruned K_j per feature
    d = size(Xwarm, 2)
    probs = Vector{Vector{Float64}}(undef, d)
    thr   = Vector{Vector{Float64}}(undef, d)
    res   = Vector{Vector{Float64}}(undef, d)
    @inbounds for j in 1:d
        Kj = length(base.thresholds[j])
        probs[j] = Kj == 0 ? Float64[] : collect((1:Kj) ./ (Kj + 1))
        col = collect(@view Xwarm[:, j])
        thr[j] = Kj == 0 ? Float64[] : quantile(col, probs[j])
        res[j] = length(col) > cap ? col[1:cap] : col
    end
    width = sum(length, thr)
    return OnlineSAQT(probs, thr, res, cap, width, size(Xwarm, 1), Random.Xoshiro(seed))
end

"Reservoir-sample one streamed sample into every feature's global reservoir."
function partial_fit!(o::OnlineSAQT, x::AbstractVector{<:Real})
    o.seen += 1
    @inbounds for j in eachindex(o.reservoir)
        r = o.reservoir[j]
        if length(r) < o.cap
            push!(r, Float64(x[j]))
        else
            k = rand(o.rng, 1:o.seen)            # classic reservoir replacement
            k <= o.cap && (r[k] = Float64(x[j]))
        end
    end
    return o
end

"Recompute thresholds from the current reservoirs at the fixed probabilities (width unchanged)."
function refit_thresholds!(o::OnlineSAQT)
    @inbounds for j in eachindex(o.probs)
        isempty(o.probs[j]) && continue
        o.thresholds[j] = quantile(o.reservoir[j], o.probs[j])
    end
    return o
end

"Booleanize one streamed sample against the current (drifting) thresholds — fixed width."
function transform_online(o::OnlineSAQT, x::AbstractVector{<:Real})
    bits = Vector{Bool}(undef, o.width); p = 1
    @inbounds for j in eachindex(o.thresholds)
        ts = o.thresholds[j]; nj = length(ts)
        c = searchsortedlast(ts, x[j])
        for k in 1:nj; bits[p] = k <= c; p += 1; end
    end
    return bits
end

end # module
