"""
    interface.jl

Interactive terminal interface: prompts the user for the contract to price
(derivative type, call/put, strike), the model inputs, and the capital to
deploy into the hedge; applies defaults (risk-free rate defaults to 0.05
when nothing is entered), re-prompts on invalid input, and runs the whole
pipeline — stock valuation summary, option valuation with the delta hedge
and replication check, path enumeration and graphs (terminal + PNG).
"""
module Interface

using ..Params: ModelParams, parse_float, parse_optional_float, parse_int,
                parse_choice
using ..Pricing: value_stock_tree, summarize
using ..Options: OptionSpec, value_option, summarize_option
using ..MonteCarlo: value_option_mc, summarize_mc, simulate_paths
using ..Paths: enumerate_paths, MAX_ENUM_PATHS
using ..Plotting: plot_price_tree, plot_paths, save_price_tree_png, save_paths_png

export ask_params, ask_contract, run

"""
    ask_contract([io_in = stdin, io_out = stdout]) -> (OptionSpec, Float64)

Prompt for the contract first — derivative type
(`european`/`american`/`lookback`/`asian`) then `call`/`put` — then the
strike `K` (skipped for lookbacks, which are floating-strike), and finally
the capital to deploy into the replicating hedge.
"""
function ask_contract(io_in::IO = stdin, io_out::IO = stdout)
    kinds = ("european", "american", "lookback", "asian")
    kind = _ask_one(io_in, io_out, "Derivative type ($(join(kinds, "/"))): ",
                    s -> parse_choice(s, "Derivative type", kinds), _ -> true)
    callput = _ask_one(io_in, io_out, "Call or put (call/put): ",
                       s -> parse_choice(s, "Option", ("call", "put")), _ -> true)
    K = kind == :lookback ? NaN :
        _ask_one(io_in, io_out, "Strike price (K): ",
                 s -> parse_float(s, "K"),
                 v -> v > 0 || "K must be strictly positive")
    capital = _ask_one(io_in, io_out, "Capital to deploy for the hedge: ",
                       s -> parse_float(s, "capital"),
                       v -> v > 0 || "capital must be strictly positive")
    return OptionSpec(kind, callput, K), capital
end

"""
    ask_params([io_in = stdin, io_out = stdout]) -> ModelParams

Prompt for `S0`, `u`, `d`, `n` and (optionally) `r`, re-prompting until every
answer parses. Leaving the risk-free-rate prompt empty accepts the default
0.05. The final parameters are validated with `validate` — a violated rule
re-runs the whole prompt loop so the user can correct the offending value.
"""
function ask_params(io_in::IO = stdin, io_out::IO = stdout)
    questions = [
        (prompt = "Stock price at time 0 (S0)",
         default = nothing,
         parser = str -> parse_float(str, "S0"),
         validate_fn = v -> v > 0 || "S0 must be strictly positive"),
        (prompt = "Up factor (u)",
         default = nothing,
         parser = str -> parse_float(str, "u"),
         validate_fn = v -> v > 0 || "u must be strictly positive"),
        (prompt = "Down factor (d)",
         default = nothing,
         parser = str -> parse_float(str, "d"),
         validate_fn = v -> v > 0 || "d must be strictly positive"),
        (prompt = "Number of periods (n)",
         default = nothing,
         parser = str -> parse_int(str, "n"),
         validate_fn = v -> v ≥ 1 || "n must be at least 1"),
        (prompt = "Risk-free rate (r)",
         default = 0.05,
         parser = str -> parse_optional_float(str, "r", 0.05),
         validate_fn = v -> v > -1 || "r must be greater than -1"),
    ]

    p = nothing
    while p === nothing
        answers = Any[]
        for q in questions
            suffix = q.default === nothing ? ": " : " [$(q.default)]: "
            v = _ask_one(io_in, io_out, q.prompt * suffix, q.parser, q.validate_fn)
            push!(answers, v)
        end
        try
            p = ModelParams(answers...)
        catch err
            err isa ArgumentError || rethrow()
            println(io_out, "  ⚠  ", sprint(showerror, err))
        end
    end
    return p::ModelParams
end

function _ask_one(io_in::IO, io_out::IO, prompt::AbstractString, parser, validate_fn)
    while true
        print(io_out, prompt)
        s = readline(io_in)
        local v
        try
            v = parser(s)
        catch err
            err isa ArgumentError || rethrow()
            println(io_out, "  ⚠  ", sprint(showerror, err))
            continue
        end
        ok = validate_fn(v)
        ok === true && return v
        println(io_out, "  ⚠  ", ok)
    end
end

"""
    run([io_in = stdin, io_out = stdout]; plot::Symbol = :both)
        -> NamedTuple

Full interactive session: ask for the contract and capital, then the model
parameters; print the stock valuation summary followed by the option block
(exact lattice/path-tree valuation with the delta hedge and replication
check, or a Monte Carlo estimate for path-dependent contracts when the
user picks `simulation` or `n` exceeds the enumeration cap); enumerate
every path-dependent price path — or draw a sample of simulated paths —
and attempt PNG output.

Returns `(stock = StockValuation, option = OptionValuation-or-nothing,
mc = MCValuation-or-nothing)`. `plot` selects terminal-only (`:terminal`),
PNG-only (`:png`) or both.
"""
function run(io_in::IO = stdin, io_out::IO = stdout; plot::Symbol = :both)
    spec, capital = ask_contract(io_in, io_out)
    p = ask_params(io_in, io_out)
    println(io_out)
    println(io_out, "Computing valuation for S0=$(p.S0), u=$(p.u), d=$(p.d), n=$(p.n), r=$(p.r) ...")

    val = value_stock_tree(p)
    summarize(io_out, val, p)
    println(io_out)

    local ov = nothing
    local mcv = nothing
    if spec.kind in (:lookback, :asian)
        local m::Int
        use_mc = false
        if p.n > MAX_ENUM_PATHS
            println(io_out, "$(spec.kind) needs the full 2^$(p.n) path tree — " *
                            "cap is n ≤ $MAX_ENUM_PATHS; switching to Monte Carlo simulation.")
            use_mc = true
        else
            method = _ask_one(io_in, io_out, "Pricing method (exact/simulation): ",
                              s -> parse_choice(s, "Pricing method", ("exact", "simulation")),
                              _ -> true)
            use_mc = method == :simulation
        end
        if use_mc
            m = _ask_one(io_in, io_out, "Monte Carlo paths (m): ",
                         s -> parse_int(s, "m"),
                         v -> v ≥ 100 || "m must be at least 100 for a meaningful estimate")
            mcv = value_option_mc(p, spec, m, capital)
            summarize_mc(io_out, mcv)
        else
            ov = value_option(p, spec, capital)
            summarize_option(io_out, ov)
        end
    else
        ov = value_option(p, spec, capital)
        summarize_option(io_out, ov)
    end
    println(io_out)

    result = (stock = val, option = ov, mc = mcv)
    n = p.n
    ps = ov !== nothing && ov.paths !== nothing ? ov.paths :
         (n ≤ MAX_ENUM_PATHS ? enumerate_paths(p; lattice = val.lattice) : nothing)

    if ps === nothing
        if mcv !== nothing && plot != :none
            sample = simulate_paths(p, min(mcv.m, 40))
            if plot in (:terminal, :both)
                println(io_out, "Simulated price paths, steps 0–$n:")
                plot_paths(io_out, sample;
                           title = "Simulated price paths (sample of $(length(sample)))")
                println(io_out)
            end
            if plot in (:png, :both)
                f = save_paths_png(sample; filename = "mc_paths.png",
                                   title = "Simulated price paths")
                if f === nothing
                    println(io_out, "PNG output skipped (Plots.jl/GR unavailable or backend failed).")
                else
                    println(io_out, "PNG written: $f")
                end
            end
        else
            println(io_out, "n = $n gives 2^$n paths — skipping explicit per-path " *
                            "enumeration (cap n ≤ $MAX_ENUM_PATHS).")
        end
        return result
    end

    println(io_out, "Enumerated $(length(ps.prices)) path-dependent price paths.")
    println(io_out)

    if plot in (:terminal, :both)
        println(io_out, "Price tree (recombining, steps 0–$n):")
        plot_price_tree(io_out, val.lattice; title = "Price tree — recombining lattice")
        println(io_out)

        println(io_out, "Every path-dependent price path, steps 0–$n:")
        plot_paths(io_out, ps; title = "All price paths — steps 0–$n")
        println(io_out)
    end

    if plot in (:png, :both)
        f1 = save_price_tree_png(val.lattice; title = "Binomial price tree (S0=$(p.S0), u=$(p.u), d=$(p.d), n=$(p.n))")
        f2 = save_paths_png(ps; title = "All $(length(ps.prices)) price paths")
        for f in (f1, f2)
            if f === nothing
                println(io_out, "PNG output skipped (Plots.jl/GR unavailable or backend failed).")
            else
                println(io_out, "PNG written: $f")
            end
        end
    end
    return result
end

end # module Interface
