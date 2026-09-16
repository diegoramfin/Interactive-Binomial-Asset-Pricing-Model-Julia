using Test
using BinomialAssetPricing
using BinomialAssetPricing.Params
using BinomialAssetPricing.Lattice
using BinomialAssetPricing.RiskNeutral
using BinomialAssetPricing.Paths
using BinomialAssetPricing.Pricing

@testset "Params" begin
    p = ModelParams(100.0, 1.2, 0.8, 3)                       # r defaults to 0.05
    @test p.r == 0.05
    @test p.n == 3
    @test risk_neutral_prob(p) ≈ ((1.05 - 0.8) / (1.2 - 0.8)) # = 0.625
    @test risk_neutral_down_prob(p) ≈ ((1.2 - 1.05) / (1.2 - 0.8)) # = 0.375
    @test risk_neutral_prob(p) + risk_neutral_down_prob(p) ≈ 1.0  # p̃ + q̃ = 1
    @test validate(p) === nothing

    # Out-of-range / malformed inputs are rejected.
    @test_throws ArgumentError validate(ModelParams(0.0, 1.2, 0.8, 3))
    @test_throws ArgumentError validate(ModelParams(100.0, 0.8, 1.2, 3))   # u ≤ d
    @test_throws ArgumentError validate(ModelParams(100.0, 1.1, 0.95, 2, 0.10)) # d < 1+r < u fails
    @test_throws ArgumentError validate(ModelParams(100.0, 1.2, 0.8, 0))
    @test_throws ArgumentError validate(ModelParams(100.0, NaN, 0.8, 3))

    # Strict parsing helpers.
    @test parse_float(" 100.5 ", "S0") == 100.5
    @test parse_int(" 5 ", "n") == 5
    @test parse_optional_float("", "r", 0.05) == 0.05        # empty → default
    @test parse_optional_float(" 0.03 ", "r", 0.05) == 0.03
    @test parse_choice(" American ", "kind", ("european", "american")) == :american
    @test_throws ArgumentError parse_float("", "S0")
    @test_throws ArgumentError parse_float("abc", "S0")
    @test_throws ArgumentError parse_float("NaN", "S0")
    @test_throws ArgumentError parse_int("3.5", "n")
    @test_throws ArgumentError parse_int("0", "n")
    @test_throws ArgumentError parse_choice("barrier", "kind", ("european", "american"))
end

@testset "Lattice" begin
    lat = build_price_lattice(100.0, 1.2, 0.8, 2)
    @test lat == [[100.0], [120.0, 80.0], [144.0, 96.0, 64.0]]
    @test length(build_price_lattice(100, 1.2, 0.8, 5)) == 6
    @test_throws ArgumentError build_price_lattice(-1.0, 1.2, 0.8, 2)

    # Risk-neutral backward induction on the stock itself reproduces S0:
    # the discounted stock is a Q-martingale.
    p = ModelParams(100.0, 1.2, 0.8, 5)
    q = risk_neutral_prob(p)
    vals = price(identity, build_price_lattice(p.S0, p.u, p.d, p.n), p.r, q)
    @test vals[1][1] ≈ 100.0 atol = 1e-8

    # European put via backward induction equals the direct expectation
    # Σ C(n,j) q^(n-j) (1-q)^j max(K - S0 u^(n-j) d^j, 0), discounted once.
    K = 105.0
    pv = price(european_put_payoff(K), build_price_lattice(100, 1.2, 0.8, 3), 0.05, 0.625)
    direct = sum(binomial(3, j) * 0.625^(3 - j) * 0.375^j *
                 max(K - 100 * 1.2^(3 - j) * 0.8^j, 0.0) for j in 0:3) / 1.05^3
    @test pv[1][1] ≈ direct atol = 1e-10

    @test_throws ArgumentError price(identity, lat, 0.05, 1.5)
end

@testset "RiskNeutral identities" begin
    for (S0, u, d, n, r) in [(100.0, 1.2, 0.8, 4, 0.05), (50.0, 1.1, 0.9, 7, 0.02)]
        p = ModelParams(S0, u, d, n, r)
        @test expected_terminal_stock(p) ≈ S0 * (1 + r)^n   # Q-martingale
        @test forward_price(p) ≈ S0 * (1 + r)^n             # cost of carry
    end

    # Recursive mass propagation: each node splits its mass into the two
    # independently derived p̃/q̃, so a valid measure keeps every tree level —
    # and the 2^n terminal paths — at sum 1.
    for n in (1, 2, 3, 10)
        sums = probability_level_sums(0.625, 0.375, n)
        @test length(sums) == n + 1
        @test all(s -> s ≈ 1.0, sums)
    end
    # Mismatched branches (p̃ + q̃ ≠ 1) must NOT conserve mass.
    @test !all(s -> s ≈ 1.0, probability_level_sums(0.6, 0.5, 3))
    @test_throws ArgumentError probability_level_sums(1.5, 0.4, 3)
    @test_throws ArgumentError probability_level_sums(0.5, -0.2, 3)
    @test_throws ArgumentError probability_level_sums(0.5, 0.5, 0)
end

@testset "Paths" begin
    p = ModelParams(100.0, 1.2, 0.8, 2)
    ps = enumerate_paths(p)
    @test length(ps.prices) == 4
    @test ps.prices[1] == [100.0, 120.0, 144.0]             # up, up
    @test ps.prices[4] == [100.0, 80.0, 64.0]               # down, down
    @test sum(ps.probs) ≈ 1.0 atol = 1e-12
    @test all(ps.prices[i][1] == 100.0 for i in 1:4)

    # Every explicit path must agree with the recombining lattice: a path's
    # node at step k is 1 + (number of down moves taken in the first k steps).
    lat = build_price_lattice(p.S0, p.u, p.d, p.n)
    for (moves, prices) in zip(ps.moves, ps.prices)
        node = 1
        @test prices[1] == lat[1][1]
        for (k, mv) in enumerate(moves)
            node += mv == 0 ? 1 : 0
            @test prices[k + 1] == lat[k + 1][node]
        end
    end

    @test_throws ArgumentError enumerate_paths(ModelParams(100, 1.2, 0.8, 25))
end

@testset "Pricing facade" begin
    p = ModelParams(100.0, 1.2, 0.8, 4)
    v = value_stock_tree(p)
    @test v.q ≈ 0.625
    @test v.S0_fair ≈ 100.0 atol = 1e-8
    @test v.forward0 ≈ 100 * 1.05^4
    @test length(v.lattice) == 5 && length(v.values) == 5
    @test all(length(v.lattice[k]) == k for k in 1:5)
    io = IOBuffer()
    summarize(io, v, p)
    s = String(take!(io))
    @test occursin("0.625", s)                              # p̃ (up prob)
    @test occursin("0.375", s)                              # q̃ (down prob)
    @test occursin("PASS", s)                               # level-mass check
end

@testset "Options" begin
    using BinomialAssetPricing.Options

    # Lecture lookback example: S0=4, u=2, d=1/2, r=1/4, n=3.
    # Floating-strike lookback put → V0 = 1.376, Δ0 = 1.04/6 ≈ 0.1733.
    p = ModelParams(4.0, 2.0, 0.5, 3, 0.25)
    ov = value_option(p, OptionSpec(:lookback, :put), 1000.0)
    @test ov.V0 ≈ 1.376 atol = 1e-10
    @test ov.deltas[1][1] ≈ 1.04 / 6 atol = 1e-10
    @test ov.replication_error ≤ 1e-10        # X_k = V_k on all 8 paths
    @test ov.contracts ≈ 1000.0 / 1.376

    # Put–call parity on the lattice: C − P = S0 − K(1+r)^{-n}.
    p2 = ModelParams(100.0, 1.2, 0.8, 4, 0.05)
    c = value_option(p2, OptionSpec(:european, :call, 105.0), 100.0)
    put = value_option(p2, OptionSpec(:european, :put, 105.0), 100.0)
    @test c.V0 - put.V0 ≈ 100.0 - 105.0 / 1.05^4 atol = 1e-10
    @test c.replication_error ≤ 1e-8
    @test c.expected_payoff ≈ c.V0 * 1.05^4    # V0 = E^Q[payoff] / (1+r)^n

    # American call on a non-dividend stock: early exercise is never
    # optimal → premium 0 and price equals the European call.
    ac = value_option(p2, OptionSpec(:american, :call, 105.0), 100.0)
    @test ac.exercise_premium ≈ 0.0 atol = 1e-10
    @test ac.V0 ≈ c.V0 atol = 1e-10

    # American put is worth strictly more here; its premium is positive.
    ap = value_option(p2, OptionSpec(:american, :put, 105.0), 100.0)
    @test ap.V0 > put.V0
    @test ap.exercise_premium > 0
    @test ap.replication_error ≤ 1e-8
    @test isfinite(ap.expected_payoff) && ap.expected_payoff > 0

    # A worthless option still prices, but capital sizing is n/a (not Inf).
    zero = value_option(p2, OptionSpec(:european, :call, 10_000.0), 100.0)
    @test zero.V0 == 0.0
    @test isnan(zero.contracts)

    # Input validation.
    @test_throws ArgumentError value_option(p2, OptionSpec(:european, :call, -1.0), 100)
    @test_throws ArgumentError value_option(p2, OptionSpec(:exotic, :call, 100.0), 100)
    @test_throws ArgumentError value_option(p2, OptionSpec(:european, :call, 100.0), -5)
    @test_throws ArgumentError value_option(p2, OptionSpec(:european, :call, 105.0), Inf)
    @test_throws ArgumentError value_option(ModelParams(100, 1.2, 0.8, 25),
                                            OptionSpec(:lookback, :put), 100)
    @test_throws ArgumentError path_payoff(OptionSpec(:exotic, :straddle, 1.0),
                                           [100.0, 120.0])

    io = IOBuffer()
    summarize_option(io, ov)
    s = String(take!(io))
    @test occursin("1.376", s)
    @test occursin("PASS", s)

    io0 = IOBuffer()
    summarize_option(io0, zero)
    s0 = String(take!(io0))
    @test occursin("n/a (V0 = 0)", s0)
    @test !occursin("Inf", s0) && !occursin("NaN", s0)
end

@testset "Interface" begin
    using BinomialAssetPricing.Interface
    using BinomialAssetPricing.Plotting

    # Contract block first (type, call/put, K, capital), then market
    # params; empty risk-free rate → default 0.05; bad values re-prompt.
    input = IOBuffer("european\ncall\n105\n1000\n100\n1.2\n0.8\n3\n\n")
    out = IOBuffer()
    res = Interface.run(input, out; plot = :none)
    s = String(take!(out))
    @test res.stock.S0_fair ≈ 100.0 atol = 1e-8
    @test res.option.V0 > 0
    @test occursin("Enumerated 8", s)                         # 2^3 paths
    @test occursin("[0.05]", s)                               # default shown in prompt
    @test occursin("PASS", s)                                 # replication check

    # Invalid entries trigger re-prompts, then succeed.
    input2 = IOBuffer("european\nput\n50\n500\n-5\n100\n1.2\n0.8\n2\n0.03\n")
    out2 = IOBuffer()
    res2 = Interface.run(input2, out2; plot = :none)
    @test res2.stock.q ≈ (1.03 - 0.8) / 0.4
    @test occursin("strictly positive", String(take!(out2)))

    # Lookback end-to-end (no strike prompt): S0=4,u=2,d=0.5,n=3,r=0.25.
    # `exact` answers the pricing-method prompt for path-dependent kinds.
    input3 = IOBuffer("lookback\nput\n100\n4\n2\n0.5\n3\n0.25\nexact\n")
    out3 = IOBuffer()
    res3 = Interface.run(input3, out3; plot = :none)
    @test res3.option.V0 ≈ 1.376 atol = 1e-10
    @test occursin("lookback put", String(take!(out3)))

    # Terminal plotting works headlessly (fallback or UnicodePlots), PNG skipped.
    io = IOBuffer()
    lat = build_price_lattice(100, 1.2, 0.8, 3)
    plot_price_tree(io, lat)
    s = String(take!(io))
    @test occursin("Underlying Prices", s)
    @test occursin("100.00", s)         # root price labeled
    @test occursin("172.80", s)         # top-most terminal price labeled
    @test occursin('/', s) && occursin('\\', s)   # tree edges present

    # Option value tree (recombining) renders with node labels.
    p_eu = ModelParams(100.0, 1.2, 0.8, 3)
    ov_eu = value_option(p_eu, OptionSpec(:european, :call, 105.0), 100.0)
    io_v = IOBuffer()
    plot_value_tree(io_v, ov_eu.values)
    sv = String(take!(io_v))
    @test occursin("Option Prices", sv)
    @test occursin("0.00", sv)          # zero-valued nodes appear

    io2 = IOBuffer()
    plot_paths(io2, enumerate_paths(ModelParams(100, 1.2, 0.8, 3)))
    @test !isempty(String(take!(io2)))

    # PNG functions degrade to `nothing` when Plots.jl is not loadable.
    if !Plotting._plots
        @test save_price_tree_png(lat) === nothing
        @test save_value_tree_png(ov_eu.values) === nothing
        @test save_paths_png(enumerate_paths(ModelParams(100, 1.2, 0.8, 2))) === nothing
    end
end

@testset "MonteCarlo" begin
    using BinomialAssetPricing.MonteCarlo
    using BinomialAssetPricing.Options
    using BinomialAssetPricing.Interface
    using Random

    # Simulated paths are honest binomial random walks: start at S0, every
    # step multiplies by u or d.
    paths = simulate_paths(MersenneTwister(7), ModelParams(100.0, 1.2, 0.8, 4), 50)
    @test length(paths) == 50
    for pr in paths
        @test length(pr) == 5
        @test pr[1] == 100.0
        @test all(pr[k + 1] == pr[k] * 1.2 || pr[k + 1] == pr[k] * 0.8 for k in 1:4)
    end

    # Same seed → identical estimate.
    p = ModelParams(4.0, 2.0, 0.5, 3, 0.25)
    spec_lb = OptionSpec(:lookback, :put)
    a = value_option_mc(p, spec_lb, 1_000, 100.0; rng = MersenneTwister(42))
    b = value_option_mc(p, spec_lb, 1_000, 100.0; rng = MersenneTwister(42))
    @test a.V0 == b.V0

    # MC lookback vs the exact path-tree price (lecture example, V0 = 1.376).
    mc = value_option_mc(p, spec_lb, 50_000, 1000.0; rng = MersenneTwister(123))
    @test abs(mc.V0 - 1.376) ≤ max(4 * mc.std_error, 0.02)

    # MC asian vs the exact path-tree price.
    p2 = ModelParams(100.0, 1.2, 0.8, 4, 0.05)
    spec_as = OptionSpec(:asian, :call, 100.0)
    exact = value_option(p2, spec_as, 100.0).V0
    mc_as = value_option_mc(p2, spec_as, 50_000, 100.0; rng = MersenneTwister(321))
    @test abs(mc_as.V0 - exact) ≤ max(4 * mc_as.std_error, 0.05)

    # MC european converges to the lattice price.
    spec_eu = OptionSpec(:european, :call, 105.0)
    eu_exact = value_option(p2, spec_eu, 100.0).V0
    mc_eu = value_option_mc(p2, spec_eu, 50_000, 100.0; rng = MersenneTwister(99))
    @test abs(mc_eu.V0 - eu_exact) ≤ max(4 * mc_eu.std_error, 0.05)

    # Analytic asian anchor: K ≈ 0 makes the max never bind, so the exact
    # price is the discounted expected average (which includes S0).
    tiny = value_option(p2, OptionSpec(:asian, :call, 0.0001), 100.0)
    avg_expectation = 100 / 5 * sum(1.05^k for k in 0:4)
    @test tiny.V0 ≈ (avg_expectation - 0.0001) / 1.05^4 atol = 1e-10

    # Validation: american rejected, m ≥ 2, asian needs a strike.
    @test_throws ArgumentError value_option_mc(p2, OptionSpec(:american, :put, 105.0),
                                               100, 100.0)
    @test_throws ArgumentError value_option_mc(p2, spec_as, 1, 100.0)
    @test_throws ArgumentError value_option_mc(p2, OptionSpec(:asian, :call), 100, 100.0)
    @test_throws ArgumentError value_option(p2, OptionSpec(:asian, :call), 100.0)

    io = IOBuffer()
    summarize_mc(io, mc)
    s = String(take!(io))
    @test occursin("Monte Carlo", s)
    @test occursin("95% confidence", s)

    # Interface: asian priced exactly when chosen.
    in1 = IOBuffer("asian\ncall\n105\n1000\n100\n1.2\n0.8\n3\n\nexact\n")
    res1 = Interface.run(in1, IOBuffer(); plot = :none)
    @test res1.option.V0 ≈ value_option(ModelParams(100.0, 1.2, 0.8, 3),
                                        OptionSpec(:asian, :call, 105.0), 1000.0).V0
    @test res1.mc === nothing

    # Interface: asian via simulation.
    in2 = IOBuffer("asian\nput\n95\n500\n100\n1.2\n0.8\n3\n0.05\nsimulation\n5000\n")
    res2 = Interface.run(in2, IOBuffer(); plot = :none)
    @test res2.mc !== nothing
    @test res2.option === nothing

    # Interface: n over the path cap forces Monte Carlo (was a dead end).
    in3 = IOBuffer("lookback\nput\n100\n4\n2\n0.5\n25\n0.25\n1000\n")
    res3 = Interface.run(in3, IOBuffer(); plot = :none)
    @test res3.mc.V0 > 0
    @test res3.option === nothing
end
