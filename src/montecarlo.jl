"""
    montecarlo.jl

Monte Carlo pricing for path-dependent options on the binomial model's own
random walk: each simulated step multiplies the price by `u` or `d` with
risk-neutral probability `q̃`, exactly the discrete geometric random walk the
lattice enumerates. Payoffs are averaged over `m` simulated paths and
discounted by `(1+r)^n`; the report carries the standard error and a 95%
confidence interval.

Forward simulation cannot do optimal stopping, so `:american` contracts are
rejected — they stay lattice-only. `:european` is accepted as a sanity check:
its Monte Carlo price converges to the exact lattice price.
"""
module MonteCarlo

using Random
using Statistics: mean, std
using ..Params: ModelParams, risk_neutral_prob, validate
using ..Options: OptionSpec, path_payoff, contract_label

export simulate_paths, value_option_mc, summarize_mc, MCValuation

"""
    simulate_paths(rng::AbstractRNG, p::ModelParams, m::Integer;
                   p_up::Real = risk_neutral_prob(p))
        -> Vector{Vector{Float64}}

Draw `m` price paths of length `n+1` (`prices[1] == S0`); each step
multiplies by `u` with probability `p_up`, else by `d`. The `p_up` kwarg
mirrors `enumerate_paths`' scenario-probability hook — pass a physical
probability to simulate under a different measure.
"""
function simulate_paths(rng::AbstractRNG, p::ModelParams, m::Integer;
                        p_up::Real = risk_neutral_prob(p))
    validate(p)
    m ≥ 1 || throw(ArgumentError("m must be at least 1 (got $m)."))
    (0 < p_up < 1) || throw(ArgumentError(
        "p_up must lie strictly between 0 and 1 (got $p_up)."))
    paths = Vector{Vector{Float64}}(undef, m)
    for j in 1:m
        prices = Vector{Float64}(undef, p.n + 1)
        _rand_path!(rng, prices, p.S0, p.u, p.d, p_up)
        paths[j] = prices
    end
    return paths
end

simulate_paths(p::ModelParams, m::Integer; kw...) =
    simulate_paths(Random.default_rng(), p, m; kw...)

# Fill `prices` in place with one random walk: S0, then *= u w.p. p_up else d.
function _rand_path!(rng::AbstractRNG, prices::Vector{Float64},
                     S0::Real, u::Real, d::Real, p_up::Real)
    prices[1] = S0
    for k in 2:length(prices)
        prices[k] = prices[k - 1] * (rand(rng) < p_up ? u : d)
    end
    return prices
end

"""
    MCValuation

Monte Carlo estimate of one `OptionSpec` + `ModelParams` + capital.

# Fields
- `spec::OptionSpec`: the contract priced.
- `m::Int`: number of simulated paths.
- `V0::Float64`: discounted mean payoff — the MC fair value.
- `std_error::Float64`: standard error of the **discounted** estimate.
- `expected_payoff::Float64`: undiscounted simulated mean `E^Q[payoff]`.
- `capital::Float64`: cash the user deploys.
- `contracts::Float64`: `capital / V0`; `NaN` when `V0 ≤ 0` (same
  convention as `OptionValuation`).
"""
struct MCValuation
    spec::OptionSpec
    m::Int
    V0::Float64
    std_error::Float64
    expected_payoff::Float64
    capital::Float64
    contracts::Float64
end

"""
    value_option_mc(p::ModelParams, spec::OptionSpec, m::Integer,
                    capital::Real; rng::AbstractRNG = Random.default_rng())
        -> MCValuation

Monte Carlo price of `:european`, `:lookback` or `:asian` contracts over
`m` simulated paths of the binomial random walk under `q̃`. `:american`
contracts need backward induction (optimal stopping) and are rejected —
use `value_option`.
"""
function value_option_mc(p::ModelParams, spec::OptionSpec, m::Integer,
                         capital::Real; rng::AbstractRNG = Random.default_rng())
    validate(p)
    spec.kind in (:european, :lookback, :asian) || throw(ArgumentError(
        "Monte Carlo pricing supports :european, :lookback and :asian " *
        "(got $(spec.kind)); american contracts need the lattice — use value_option."))
    spec.callput in (:call, :put) || throw(ArgumentError(
        "Option must be :call or :put (got $(spec.callput))."))
    (isfinite(capital) && capital > 0) || throw(ArgumentError(
        "Capital must be a finite, strictly positive number (got $capital)."))
    m ≥ 2 || throw(ArgumentError(
        "m must be at least 2 (standard error needs ≥ 2 paths, got $m)."))
    if spec.kind != :lookback
        (isfinite(spec.K) && spec.K > 0) || throw(ArgumentError(
            "European/American/Asian options need a strictly positive strike K " *
            "(got $(spec.K))."))
    end

    q = risk_neutral_prob(p)
    prices = Vector{Float64}(undef, p.n + 1)
    payoffs = Vector{Float64}(undef, m)
    for j in 1:m
        _rand_path!(rng, prices, p.S0, p.u, p.d, q)
        payoffs[j] = path_payoff(spec, prices)
    end

    disc = (1 + p.r)^p.n
    expected_payoff = mean(payoffs)
    V0 = expected_payoff / disc
    std_error = std(payoffs) / sqrt(m) / disc
    contracts = V0 > 0 ? capital / V0 : NaN
    return MCValuation(spec, Int(m), V0, std_error, expected_payoff,
                       Float64(capital), contracts)
end

"""
    summarize_mc(io::IO, mc::MCValuation)

Print the Monte Carlo report: contract label, MC fair value with its
standard error and 95% confidence interval, the simulated mean payoff,
and the contract count implied by the deployed capital. No delta or
replication lines — Monte Carlo prices but does not hedge.
"""
function summarize_mc(io::IO, mc::MCValuation)
    lo = mc.V0 - 1.96 * mc.std_error
    hi = mc.V0 + 1.96 * mc.std_error
    println(io, "─"^58)
    println(io, " Option: $(contract_label(mc.spec))  ·  Monte Carlo (m=$(mc.m) paths)")
    println(io, "─"^58)
    println(io, " ", rpad("MC fair value V0", 35), ": ", round(mc.V0, digits = 6))
    println(io, " ", rpad("Std error of V0", 35), ": ", round(mc.std_error, digits = 6))
    println(io, " ", rpad("95% confidence interval", 35), ": ",
            "[$(round(lo, digits = 6)), $(round(hi, digits = 6))]")
    println(io, " ", rpad("E^Q[payoff] (simulated mean)", 35), ": ",
            round(mc.expected_payoff, digits = 6))
    println(io, " ", rpad("Capital deployed", 35), ": ", round(mc.capital, digits = 6))
    if isnan(mc.contracts)
        println(io, " ", rpad("Contracts (capital / V0)", 35), ": n/a (V0 = 0)")
    else
        println(io, " ", rpad("Contracts (capital / V0)", 35), ": ",
                round(mc.contracts, digits = 6))
    end
    println(io, "─"^58)
    return nothing
end

summarize_mc(mc::MCValuation) = summarize_mc(stdout, mc)

end # module MonteCarlo
