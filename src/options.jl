"""
    options.jl

Derivative pricing, delta hedging and replication on the binomial tree.

Four contract kinds, all priced under the risk-neutral measure
`p̃ = (1+r−d)/(u−d)`, `q̃ = 1−p̃`:

- `:european` — backward induction of `payoff(S_n)` on the recombining
  lattice ([`Lattice.price`](@ref)).
- `:american` — same, but each node takes `max(exercise, continuation)`
  ([`Lattice.price_american`](@ref)).
- `:lookback` — floating-strike path-dependent claim on the full `2^n`
  path tree (call: `S_n − min S`, put: `max S − S_n`), exactly the
  lookback example in the lecture notes.
- `:asian` — fixed-strike arithmetic-average claim on the same path
  tree (call: `max(avg − K, 0)`, put: `max(K − avg, 0)`), where `avg`
  is the mean of all `n+1` path prices including `S0`.

For every contract the module also computes the delta hedge

    Δ_n(ω) = [V_{n+1}(ωH) − V_{n+1}(ωT)] / [S_{n+1}(ωH) − S_{n+1}(ωT)]

and rolls the replicating portfolio forward via the wealth equation
`X_{n+1} = Δ_n S_{n+1} + (1+r)(X_n − Δ_n S_n)`, checking `X_k(ω) = V_k(ω)`
on every path and at every step — Theorem 1.2.2's replication guarantee.
For `:american` contracts replication holds up to the optimal exercise
(stopping) time: once `exercise > continuation` at a node, the contract
is settled and the proceeds sit in the money market.
"""
module Options

using ..Params: ModelParams, risk_neutral_prob, validate
using ..Lattice: build_price_lattice, price, price_american
using ..RiskNeutral: european_call_payoff, european_put_payoff
using ..Paths: PathSet, enumerate_paths, MAX_ENUM_PATHS

export OptionSpec, OptionValuation, value_option, summarize_option, path_payoff,
       contract_label

"""
    OptionSpec

Which derivative to price.

# Fields
- `kind::Symbol`: `:european`, `:american`, `:lookback` or `:asian`.
- `callput::Symbol`: `:call` or `:put`. For `:lookback` these are the
  floating-strike versions — call pays `S_n − min_k S_k`, put pays
  `max_k S_k − S_n`. For `:asian` they are the fixed-strike
  arithmetic-average versions — call pays `max(avg − K, 0)`, put pays
  `max(K − avg, 0)` with `avg` the mean of the whole path.
- `K::Float64`: strike; ignored (`NaN`) only for `:lookback`.

`OptionSpec(kind, callput)` builds a strikeless spec (lookback).
"""
struct OptionSpec
    kind::Symbol
    callput::Symbol
    K::Float64
end

OptionSpec(kind::Symbol, callput::Symbol) = OptionSpec(kind, callput, NaN)

"""
    OptionValuation

Everything computed for one `OptionSpec` + `ModelParams` + capital.

# Fields
- `spec::OptionSpec`: the contract priced.
- `lattice::Vector{Vector{Float64}}`: recombining stock price lattice.
- `values::Vector{Vector{Float64}}`: option value tree `V_k`; row `k+1`
  holds `k+1` lattice nodes for european/american, or `2^k` path
  histories for lookback (history `h`, bit `i` = toss `i+1`, 0 = up).
- `deltas::Vector{Vector{Float64}}`: `deltas[k]` = `Δ_{k-1}` per
  node/history, levels `0…n-1`.
- `exercise_nodes::Union{Nothing,Vector{BitVector}}`: `:american` only —
  `exercise_nodes[k+1][i]` marks nodes where immediate exercise beats
  continuation (optimal stopping); `nothing` otherwise.
- `V0::Float64`: time-0 fair value.
- `expected_payoff::Float64`: `E^Q[payoff]` undiscounted — the terminal
  payoff for european/lookback contracts, the payoff at the optimal
  exercise time for `:american`; `NaN` when `n` exceeds the
  path-enumeration cap.
- `exercise_premium::Float64`: `:american` only — `V0` minus the matching
  european `V0`; `NaN` otherwise.
- `capital::Float64`: cash the user deploys into the hedge.
- `contracts::Float64`: `capital / V0` — how many contracts the capital
  replicates.
- `paths::Union{PathSet,Nothing}`: enumerated `2^n` paths (`nothing` when
  `n > MAX_ENUM_PATHS` for lattice-priced contracts).
- `wealth::Union{Nothing,Vector{Vector{Float64}}}`: per-path replicating
  portfolio values `X_0…X_n` (per contract), retained only when `n ≤ 6`
  so the wealth table can be printed; `nothing` otherwise — the
  replication check streams without storing at larger `n`.
- `replication_error::Float64`: `max |X_k(ω) − V_k(ω)|` over all paths
  and steps (up to the exercise time for `:american`); `NaN` when the
  pathwise check was skipped.
"""
struct OptionValuation
    spec::OptionSpec
    lattice::Vector{Vector{Float64}}
    values::Vector{Vector{Float64}}
    deltas::Vector{Vector{Float64}}
    exercise_nodes::Union{Nothing,Vector{BitVector}}
    V0::Float64
    expected_payoff::Float64
    exercise_premium::Float64
    capital::Float64
    contracts::Float64
    paths::Union{PathSet,Nothing}
    wealth::Union{Nothing,Vector{Vector{Float64}}}
    replication_error::Float64
end

"""
    path_payoff(spec::OptionSpec, prices::Vector{Float64}) -> Float64

Terminal payoff of the contract along one price path `prices`
(`prices[1] == S0`, `prices[end] == S_n`). For `:lookback` and `:asian`
the whole path matters; vanilla contracts only read `prices[end]`.
"""
function path_payoff(spec::OptionSpec, prices::Vector{Float64})
    spec.kind in (:european, :american, :lookback, :asian) || throw(ArgumentError(
        "Derivative kind must be :european, :american, :lookback or :asian (got $(spec.kind))."))
    spec.callput in (:call, :put) || throw(ArgumentError(
        "Option must be :call or :put (got $(spec.callput))."))
    if spec.kind == :lookback
        return spec.callput == :put ?
               maximum(prices) - prices[end] :
               prices[end] - minimum(prices)
    end
    if spec.kind == :asian
        avg = sum(prices) / length(prices)
        return spec.callput == :call ? max(avg - spec.K, 0.0) : max(spec.K - avg, 0.0)
    end
    return _exercise_value(spec, prices[end])
end

# Immediate-exercise value of a vanilla contract at price `S`.
_exercise_value(spec::OptionSpec, S::Real) =
    spec.callput == :call ? max(S - spec.K, 0.0) : max(spec.K - S, 0.0)

"""
    value_option(p::ModelParams, spec::OptionSpec, capital::Real)
        -> OptionValuation

Price the contract, compute the per-node delta hedge, and roll the
replicating portfolio forward along every path to verify
`X_k(ω) = V_k(ω)` (up to the exercise time for `:american`).

Lookbacks and asians are priced on the full `2^n` path tree and therefore
require `n ≤ MAX_ENUM_PATHS`; european/american contracts price on the
lattice for any `n`, but the pathwise replication check is skipped when
`n > MAX_ENUM_PATHS`.
"""
function value_option(p::ModelParams, spec::OptionSpec, capital::Real)
    validate(p)
    spec.kind in (:european, :american, :lookback, :asian) || throw(ArgumentError(
        "Derivative kind must be :european, :american, :lookback or :asian (got $(spec.kind))."))
    spec.callput in (:call, :put) || throw(ArgumentError(
        "Option must be :call or :put (got $(spec.callput))."))
    (isfinite(capital) && capital > 0) || throw(ArgumentError(
        "Capital must be a finite, strictly positive number (got $capital)."))

    lattice = build_price_lattice(p.S0, p.u, p.d, p.n)
    q = risk_neutral_prob(p)
    pathtree = spec.kind in (:lookback, :asian)
    if spec.kind != :lookback
        (isfinite(spec.K) && spec.K > 0) || throw(ArgumentError(
            "European/American/Asian options need a strictly positive strike K " *
            "(got $(spec.K))."))
    end
    exercise_nodes = nothing

    if pathtree
        p.n ≤ MAX_ENUM_PATHS || throw(ArgumentError(
            "Path-dependent options (lookback/asian) need the full 2^n path " *
            "tree — cap is n ≤ $MAX_ENUM_PATHS (got $(p.n))."))
        ps = enumerate_paths(p; lattice = lattice)
        term = [path_payoff(spec, pr) for pr in ps.prices]
        values = _pathtree_value_tree(term, p.n, p.r, q)
        deltas = _delta_pathtree(values, lattice, p.n)
        expected = sum(ps.probs .* term)
        premium = NaN
    else
        payoff = spec.callput == :call ?
                 european_call_payoff(spec.K) : european_put_payoff(spec.K)
        values = spec.kind == :american ?
                 price_american(payoff, lattice, p.r, q) :
                 price(payoff, lattice, p.r, q)
        deltas = _delta_lattice(values, lattice, p.n)
        if spec.kind == :american
            premium = values[1][1] - price(payoff, lattice, p.r, q)[1][1]
            exercise_nodes = _exercise_nodes(spec, values, lattice, p.r, q)
        else
            premium = NaN
        end
        if p.n ≤ MAX_ENUM_PATHS
            ps = enumerate_paths(p; lattice = lattice)
            term = [path_payoff(spec, pr) for pr in ps.prices]
            # American exercise is path-dependent, so the undiscounted
            # expectation is taken at the optimal exercise time of each path.
            expected = spec.kind == :american ?
                _expected_exercise_payoff(spec, exercise_nodes, ps) :
                sum(ps.probs .* term)
        else
            ps, expected = nothing, NaN
        end
    end

    V0 = values[1][1]
    if ps === nothing
        wealth, rep_err = nothing, NaN
    else
        # Wealth series are only worth retaining when the table is printable.
        wealth, rep_err = _replicate_wealth(values, deltas, exercise_nodes,
                                            ps, p.r, spec; store = p.n ≤ 6)
    end
    # A worthless contract has no meaningful contract count — report NaN
    # rather than Inf so the summary can label it.
    contracts = V0 > 0 ? capital / V0 : NaN
    return OptionValuation(spec, lattice, values, deltas, exercise_nodes,
                           V0, expected, premium, Float64(capital),
                           contracts, ps, wealth, rep_err)
end

# ------------------------------------------------------------------
# Internals
# ------------------------------------------------------------------

# Nodes where immediate exercise strictly beats the discounted
# continuation value — i.e. where the american value tree chose the
# payoff. `result[k+1][i]` covers levels k = 0…n-1.
function _exercise_nodes(spec::OptionSpec, values, lattice, r::Real, q::Real)
    n = length(lattice) - 1
    disc = 1 / (1 + r)
    flags = Vector{BitVector}(undef, n)
    for k in 0:(n - 1)
        row = falses(k + 1)
        v, s = values[k + 2], lattice[k + 1]
        for i in 1:(k + 1)
            cont = disc * (q * v[i] + (1 - q) * v[i + 1])
            row[i] = _exercise_value(spec, s[i]) > cont
        end
        flags[k + 1] = row
    end
    return flags
end

# Full non-recombining value tree for a path-dependent claim.
# `term[j+1]` is the payoff of path `j` (bit i of j = toss i+1, 0 = up).
# History h at level k has children h (up) and h + 2^k (down).
function _pathtree_value_tree(term::Vector{Float64}, n::Int, r::Real, q::Real)
    disc = 1 / (1 + r)
    values = Vector{Vector{Float64}}(undef, n + 1)
    values[n + 1] = term
    for k in (n - 1):-1:0
        nxt = values[k + 2]
        row = Vector{Float64}(undef, 2^k)
        for h in 0:(2^k - 1)
            @inbounds row[h + 1] =
                disc * (q * nxt[h + 1] + (1 - q) * nxt[h + 2^k + 1])
        end
        values[k + 1] = row
    end
    return values
end

# Δ_k(i) on the recombining lattice: children of node (k,i) are
# (k+1,i) up and (k+1,i+1) down.
function _delta_lattice(values, lattice, n::Int)
    deltas = Vector{Vector{Float64}}(undef, n)
    for k in 0:(n - 1)
        row = Vector{Float64}(undef, k + 1)
        v, s = values[k + 2], lattice[k + 2]
        for i in 1:(k + 1)
            @inbounds row[i] = (v[i] - v[i + 1]) / (s[i] - s[i + 1])
        end
        deltas[k + 1] = row
    end
    return deltas
end

# Δ_k(h) on the path tree: history h at level k (bit i = toss i+1,
# 0 = up). Node price is lattice[k+1][count_ones(h)+1]; the up child keeps
# the down-count, the down child adds one.
function _delta_pathtree(values, lattice, n::Int)
    deltas = Vector{Vector{Float64}}(undef, n)
    for k in 0:(n - 1)
        row = Vector{Float64}(undef, 2^k)
        v, s = values[k + 2], lattice[k + 2]
        for h in 0:(2^k - 1)
            downs = count_ones(h)
            @inbounds row[h + 1] =
                (v[h + 1] - v[h + 2^k + 1]) / (s[downs + 1] - s[downs + 2])
        end
        deltas[k + 1] = row
    end
    return deltas
end

# E^Q[payoff at optimal exercise] for an american contract: each path's
# contribution is the exercise value at its first flagged exercise node,
# or the terminal payoff when exercise never becomes strictly optimal.
function _expected_exercise_payoff(spec::OptionSpec,
                                   exercise_nodes::Vector{BitVector},
                                   ps::PathSet)
    n = length(ps.moves[1])
    total = 0.0
    for j in 0:(2^n - 1)
        pay = path_payoff(spec, ps.prices[j + 1])   # terminal fallback
        for k in 0:(n - 1)
            i = count_ones(j & (2^k - 1)) + 1
            if exercise_nodes[k + 1][i]
                pay = _exercise_value(spec, ps.prices[j + 1][k + 1])
                break
            end
        end
        total += ps.probs[j + 1] * pay
    end
    return total
end

# Roll X_{k+1} = Δ_k S_{k+1} + (1+r)(X_k − Δ_k S_k) forward along every
# enumerated path; returns the stored per-path wealth series (or nothing
# when `store` is false) and the worst |X_k(ω) − V_k(ω)| over all paths
# and steps. For american contracts the hedge is compared only until the
# optimal exercise node — after that the contract is settled and the
# proceeds just grow at the risk-free rate.
function _replicate_wealth(values, deltas,
                           exercise_nodes::Union{Nothing,Vector{BitVector}},
                           ps::PathSet, r::Real, spec::OptionSpec;
                           store::Bool)
    n = length(ps.moves[1])
    pathtree = spec.kind in (:lookback, :asian)
    wealth = store ? Vector{Vector{Float64}}(undef, 2^n) : nothing
    max_err = 0.0
    for j in 0:(2^n - 1)
        X = Vector{Float64}(undef, n + 1)
        X[1] = values[1][1]
        prices = ps.prices[j + 1]
        alive = exercise_nodes === nothing || !exercise_nodes[1][1]
        for k in 1:n
            if !alive
                store && (X[k + 1] = (1 + r) * X[k])  # exercised — cash sits
                continue
            end
            h_prev = j & (2^(k - 1) - 1)             # tosses before step k
            i_delta = pathtree ? h_prev + 1 : count_ones(h_prev) + 1
            Δ = deltas[k][i_delta]
            X[k + 1] = Δ * prices[k + 1] + (1 + r) * (X[k] - Δ * prices[k])
            h_now = j & (2^k - 1)                    # tosses through step k
            i_now = pathtree ? h_now + 1 : count_ones(h_now) + 1
            err = abs(X[k + 1] - values[k + 1][i_now])
            err > max_err && (max_err = err)
            if exercise_nodes !== nothing && k < n
                alive = !exercise_nodes[k + 1][i_now]
            end
        end
        store && (wealth[j + 1] = X)
    end
    return wealth, max_err
end

# Toss history h at level k as an H/T string: bit i of h is toss i+1
# (0 = H). Used to label lookback delta rows.
_history_label(h::Integer, k::Int) =
    join(iszero((h >> i) & 1) ? 'H' : 'T' for i in 0:(k - 1))

"""
    summarize_option(io::IO, ov::OptionValuation)

Print the option report: contract label, fair value, hedge size implied
by the deployed capital, the per-node delta tree (`*` marks optimal
early exercise for american contracts), and the per-path wealth table
showing the replication matching `V_k` step by step.
"""
function summarize_option(io::IO, ov::OptionValuation)
    spec = ov.spec
    n = length(ov.lattice) - 1
    pathtree = spec.kind in (:lookback, :asian)

    label = contract_label(spec)

    println(io, "─"^58)
    println(io, " Option: $label")
    println(io, "─"^58)
    println(io, " Option fair value V0               : $(round(ov.V0, digits=6))")
    if spec.kind == :american
        println(io, " Early-exercise premium (vs Euro.)  : " *
                    "$(round(ov.exercise_premium, digits=6))")
    end
    ep_label = spec.kind == :american ? "E^Q[payoff at exercise]" :
                                        "E^Q[payoff] (undiscounted)"
    if isnan(ov.expected_payoff)
        println(io, " ", rpad(ep_label, 35), ": n/a (2^$n paths > cap)")
    else
        println(io, " ", rpad(ep_label, 35), ": ",
                round(ov.expected_payoff, digits = 6))
    end
    println(io, " Capital deployed                   : $(round(ov.capital, digits=6))")
    if isnan(ov.contracts)
        println(io, " Contracts replicated (capital / V0): n/a (V0 = 0)")
    else
        println(io, " Contracts replicated (capital / V0): " *
                    "$(round(ov.contracts, digits=6))")
        if !isnan(ov.expected_payoff)
            println(io, " Expected payoff × contracts        : " *
                        "$(round(ov.contracts * ov.expected_payoff, digits=6))")
        end
    end
    if ov.exercise_nodes !== nothing && ov.exercise_nodes[1][1]
        println(io, " Optimal action at t=0              : exercise immediately")
        println(io, " Shares to hold now                 : 0.0 (contract settles)")
    else
        println(io, " Initial hedge Δ0 (per contract)    : " *
                    "$(round(ov.deltas[1][1], digits=6))")
        if !isnan(ov.contracts)
            println(io, " Shares to hold now (N · Δ0)        : " *
                        "$(round(ov.contracts * ov.deltas[1][1], digits=6))")
        end
    end
    if isnan(ov.replication_error)
        println(io, " Replication X_k = V_k              : skipped (2^$n > path cap)")
    else
        ok = ov.replication_error ≤ 1e-8
        detail = spec.kind == :american ?
            "up to exercise, all 2^$n paths" : "all 2^$n paths, every step"
        println(io, " Replication X_k = V_k              : " *
                    "$(ok ? "PASS" : "FAIL") (max err " *
                    "$(round(ov.replication_error, sigdigits=3)), $detail)")
    end
    println(io, "─"^58)

    println(io, " Delta hedge Δ_k (shares per contract):")
    for k in 0:(n - 1)
        row = ov.deltas[k + 1]
        shown = min(length(row), 8)
        parts = map(1:shown) do i
            s = string(round(row[i], digits = 4))
            if ov.exercise_nodes !== nothing && ov.exercise_nodes[k + 1][i]
                s *= "*"                     # optimal early exercise here
            end
            (pathtree && 0 < k && length(row) ≤ 8) ?
                "$(_history_label(i - 1, k)):$s" : s
        end
        length(row) > shown && push!(parts, "…")
        println(io, "  k=$k:  ", join(parts, "   "))
    end
    if ov.exercise_nodes !== nothing
        println(io, "  (* = early exercise strictly optimal — contract settles, " *
                    "hedge unwinds)")
    end
    println(io)

    if ov.wealth === nothing
        isnan(ov.replication_error) || println(io,
            " (per-path wealth table skipped for n > 6; the check above " *
            "covers all 2^$n paths)")
        return nothing
    end

    println(io, " Portfolio wealth X_k per path (per contract):")
    println(io, "   ω       ", join(lpad("X$k", 9) for k in 0:n), " │", lpad("V$n", 9))
    shown = min(length(ov.paths.moves), 16)
    for j in 1:shown
        ω = join(mv == 1 ? 'H' : 'T' for mv in ov.paths.moves[j])
        if ov.exercise_nodes !== nothing && _path_exercised(ov, j - 1)
            ω *= "*"                       # hit an early-exercise node
        end
        xs = join(lpad(string(round(x, digits = 4)), 9) for x in ov.wealth[j])
        vn = pathtree ? ov.values[n + 1][j] :
                        path_payoff(spec, ov.paths.prices[j])
        println(io, "   ", rpad(ω, 8), xs, " │", lpad(string(round(vn, digits = 4)), 9))
    end
    if ov.exercise_nodes !== nothing
        println(io, "  (* = option exercised early; later X is the payout in cash)")
    end
    length(ov.paths.moves) > shown &&
        println(io, "   … ($(length(ov.paths.moves) - shown) more paths)")
    return nothing
end

# Did path `j` (0-indexed) hit an early-exercise node before maturity?
function _path_exercised(ov::OptionValuation, j::Int)
    n = length(ov.lattice) - 1
    for k in 0:(n - 1)
        i = count_ones(j & (2^k - 1)) + 1
        ov.exercise_nodes[k + 1][i] && return true
    end
    return false
end

summarize_option(ov::OptionValuation) = summarize_option(stdout, ov)

"""
    contract_label(spec::OptionSpec) -> String

One-line human description of the contract, shared by the exact and
Monte Carlo report blocks.
"""
function contract_label(spec::OptionSpec)
    if spec.kind == :lookback
        pf = spec.callput == :put ? "max S − S_T" : "S_T − min S"
        return "lookback $(spec.callput) (floating strike, payoff = $pf)"
    elseif spec.kind == :asian
        pf = spec.callput == :call ? "(avg S − K)⁺" : "(K − avg S)⁺"
        return "asian $(spec.callput), K=$(spec.K) (payoff = $pf, avg over S0…Sn)"
    end
    return "$(spec.kind) $(spec.callput), K=$(spec.K)"
end

end # module Options
