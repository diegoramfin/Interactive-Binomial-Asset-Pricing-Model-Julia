"""
    plotting.jl

Graphing of price trees and individual price paths, up to and including the
final period, in two flavors:

- `:terminal` — a Unicode rendering drawn straight into any `IO` (zero setup,
  works over SSH). Drawn with `UnicodePlots` if available, else with plain
  box-drawing fallback art.
- `:png` — a vector-style PNG written to `output/` via `Plots.jl` + GR.

Everything degrades gracefully: if a backend is not installed the offending
flavor prints a short notice instead of failing the run.
"""
module Plotting

using Printf
using ..Params: ModelParams
using ..Paths: PathSet

export PlotFlavor, plot_price_tree, plot_value_tree, plot_paths,
       save_price_tree_png, save_value_tree_png, save_paths_png

"""Graph output flavor: `nothing` (skip), `:terminal`, or `:png`."""
const PlotFlavor = Union{Nothing, Symbol}

# --------------------------------------------------------------------------
# Optional backends (graceful degradation)
# --------------------------------------------------------------------------
const _unicode = let ok = true
    try
        @eval import UnicodePlots
    catch
        ok = false
    end
    ok
end

const _plots = let ok = true
    try
        @eval import Plots
    catch
        ok = false
    end
    ok
end

const PNG_DIR = joinpath(@__DIR__, "..", "output")

"Simple deterministic rainbow over `n` series (SVG-friendly color names)."
function _palette(n::Int)
    base = [:red, :orange, :goldenrod2, :green, :cyan3,
            :blue, :purple, :magenta, :deeppink, :brown]
    n ≤ 0 && return String[]
    return [base[mod1(i, length(base))] for i in 1:n]
end

# --------------------------------------------------------------------------
# Terminal plots
# --------------------------------------------------------------------------

"""
    plot_price_tree(io::IO, lattice; title = "Underlying Prices")

Draw the recombining lattice as a node-labeled tree — every node shows its
price, children connect to their parent with `/` and `\\` edges, highest
price on the right. Pure ASCII; no plotting backend needed. For `n > 12`
the labeled tree is too wide to read, so it falls back to the
`UnicodePlots` line view (or a compact table without it).
"""
function plot_price_tree(io::IO, lattice::Vector{Vector{Float64}};
                         title::AbstractString = "Underlying Prices")
    n = length(lattice) - 1
    if n ≤ 12
        _ascii_tree(io, lattice, title)
        return nothing
    end
    if !_unicode
        _fallback_tree(io, lattice, title)
        return nothing
    end
    series = collect(lattice)
    all_y = reduce(vcat, series)
    ylim = (minimum(all_y), maximum(all_y))
    plt = UnicodePlots.lineplot(_xs(series[1]), series[1];
                                title = title,
                                xlabel = "time (normalized)",
                                ylabel = "price",
                                width = min(80, displaysize(io)[2] - 4),
                                height = 16,
                                ylim = ylim)
    for s in series[2:end]
        UnicodePlots.lineplot!(plt, _xs(s), s)
    end
    show(io, plt)
    return nothing
end

"""
    plot_value_tree(io::IO, values; title = "Option Prices")

Draw a recombining option value tree in the same node-labeled style as
`plot_price_tree`. Only meaningful for `:european`/`:american` contracts —
path-dependent value trees do not recombine.
"""
function plot_value_tree(io::IO, values::Vector{Vector{Float64}};
                         title::AbstractString = "Option Prices")
    n = length(values) - 1
    if n ≤ 12
        _ascii_tree(io, values, title)
        return nothing
    end
    if !_unicode
        _fallback_tree(io, values, title)
        return nothing
    end
    series = collect(values)
    all_y = reduce(vcat, series)
    ylim = (minimum(all_y), maximum(all_y))
    plt = UnicodePlots.lineplot(_xs(series[1]), series[1];
                                title = title,
                                xlabel = "time (normalized)",
                                ylabel = "value",
                                width = min(80, displaysize(io)[2] - 4),
                                height = 16,
                                ylim = ylim)
    for s in series[2:end]
        UnicodePlots.lineplot!(plt, _xs(s), s)
    end
    show(io, plt)
    return nothing
end

# Node-labeled ASCII tree: row k sits on its own line, parents centered
# over their children, `/`/`\\` connectors on the line between levels.
# Node (k, i) — i = number of down moves — is drawn at column index
# k − i so prices increase left to right (highest on the right).
function _ascii_tree(io::IO, rows::Vector{Vector{Float64}}, title::AbstractString)
    n = length(rows) - 1
    labels = [[@sprintf("%.2f", v) for v in row] for row in rows]
    maxlen = maximum(length, Iterators.flatten(labels); init = 1)
    w = max(maxlen + 2, 8)                       # cell width per bottom node
    center(k, i) = (k - i + (n - k) / 2) * w + w / 2   # 1-based char index
    width = Int(ceil((n + 1) * w)) + 1

    println(io, title)
    for k in 0:n
        line = fill(' ', width)
        for i in 0:k
            lab = labels[k + 1][i + 1]
            c = Int(round(center(k, i)))
            start = max(c - (length(lab) - 1) ÷ 2, 1)
            for (j, ch) in enumerate(lab)
                pos = start + j - 1
                pos ≤ width && (line[pos] = ch)
            end
        end
        println(io, rstrip(String(line)))
        k == n && break
        conn = fill(' ', width)
        for i in 0:k
            c = Int(round(center(k, i)))
            lpos = Int(round(c - w / 4))         # down child is to the left
            rpos = Int(round(c + w / 4))         # up child to the right
            1 ≤ lpos ≤ width && (conn[lpos] = '/')
            1 ≤ rpos ≤ width && (conn[rpos] = '\\')
        end
        println(io, rstrip(String(conn)))
    end
    return nothing
end

"""x-coordinates for a generation of `m` nodes stretched over [0, 1].
`m == 1` (the tree root) needs an explicit single point, since
`range(0, 1; length = 1)` is contradictory in Julia."""
_xs(row::Vector{Float64}) =
    length(row) == 1 ? [0.0] : collect(range(0.0, 1.0; length = length(row)))

"""
    plot_paths(io::IO, paths; title = "Price paths", max_paths = 40)

Draw every price path in `paths` (capped at `max_paths` for legibility)
through the final period. Accepts any vector of price vectors — enumerated
`PathSet` paths or Monte Carlo simulations alike. Requires `UnicodePlots`;
falls back to ASCII art.
"""
function plot_paths(io::IO, paths::AbstractVector{<:AbstractVector{<:Real}};
                    title::AbstractString = "Price paths",
                    max_paths::Int = 40)
    isempty(paths) && return nothing
    shown = length(paths) ≤ max_paths ? paths : paths[1:max_paths]
    if !_unicode
        _fallback_paths(io, shown, title)
        return nothing
    end
    all_y = reduce(vcat, shown)
    plt = UnicodePlots.lineplot(shown[1];
                                title = title,
                                xlabel = "period",
                                ylabel = "price",
                                width = min(80, displaysize(io)[2] - 4),
                                height = 18,
                                ylim = (minimum(all_y), maximum(all_y)))
    for path in shown[2:end]
        UnicodePlots.lineplot!(plt, path)
    end
    show(io, plt)
    println(io)
    if length(paths) > max_paths
        println(io, "  (showing first $max_paths of $(length(paths)) paths)")
    end
    return nothing
end

plot_paths(io::IO, ps::PathSet; kw...) = plot_paths(io, ps.prices; kw...)

"""
    save_price_tree_png(lattice; filename = "price_tree.png", title = "Binomial price tree") -> Union{String, Nothing}

Write the recombining price tree to `output/<filename>` using `Plots.jl`:
every parent–child edge is drawn and each node is labeled with its price
(labels skipped for `n > 12` where they'd overlap). Returns the file path,
or `nothing` when `Plots.jl`/GR is unavailable.
"""
function save_price_tree_png(lattice::Vector{Vector{Float64}};
                             filename::AbstractString = "price_tree.png",
                             title::AbstractString = "Binomial price tree")
    return _tree_png(lattice, filename, title, "price")
end

"""
    save_value_tree_png(values; filename = "value_tree.png", title = "Option value tree") -> Union{String, Nothing}

Write a recombining option value tree to `output/<filename>` in the same
edge-and-label style as `save_price_tree_png`. Only meaningful for
`:european`/`:american` contracts. Returns the file path, or `nothing`
when `Plots.jl`/GR is unavailable.
"""
function save_value_tree_png(values::Vector{Vector{Float64}};
                             filename::AbstractString = "value_tree.png",
                             title::AbstractString = "Option value tree")
    return _tree_png(values, filename, title, "value")
end

# Shared PNG renderer for recombining trees: edges between every parent
# and its two children, a marker per node, and a "%.2f" label above each
# node when the tree is small enough to stay legible.
function _tree_png(rows::Vector{Vector{Float64}}, filename::AbstractString,
                   title::AbstractString, ylabel::AbstractString)
    _plots || return nothing
    n = length(rows) - 1
    plt = Plots.plot(; title = title, xlabel = "period", ylabel = ylabel,
                     legend = false, size = (800, 520), dpi = 150)
    xs = Float64[]
    ys = Float64[]
    for k in 0:(n - 1)
        for i in 1:(k + 1)
            Plots.plot!(plt, [k, k + 1], [rows[k + 1][i], rows[k + 2][i]];
                        color = :steelblue, linewidth = 1.2)
            Plots.plot!(plt, [k, k + 1], [rows[k + 1][i], rows[k + 2][i + 1]];
                        color = :steelblue, linewidth = 1.2)
        end
    end
    for k in 0:n
        append!(xs, fill(Float64(k), k + 1))
        append!(ys, rows[k + 1])
    end
    Plots.scatter!(plt, xs, ys; color = :steelblue, markersize = 3)
    if n ≤ 12
        for k in 0:n, i in 1:(k + 1)
            Plots.annotate!(plt, k, rows[k + 1][i],
                            Plots.text(@sprintf("%.2f", rows[k + 1][i]), 7,
                                       :bottom, :center))
        end
    end
    return _write_png(plt, filename)
end

"""
    save_paths_png(paths; filename = "price_paths.png", title = "Price paths", max_paths = 200) -> Union{String, Nothing}

Write every price path in `paths` (capped at `max_paths`) to
`output/<filename>` — enumerated `PathSet` paths or Monte Carlo
simulations alike. Returns the file path, or `nothing` when `Plots.jl`/GR
is unavailable.
"""
function save_paths_png(paths::AbstractVector{<:AbstractVector{<:Real}};
                        filename::AbstractString = "price_paths.png",
                        title::AbstractString = "Price paths", max_paths::Int = 200)
    _plots || return nothing
    paths = length(paths) ≤ max_paths ? paths : paths[1:max_paths]
    n = length(paths[1]) - 1
    colors = _palette(length(paths))
    plt = Plots.plot(; title = title, xlabel = "period", ylabel = "price",
                     legend = false, size = (800, 520), dpi = 150)
    for (i, path) in enumerate(paths)
        Plots.plot!(plt, 0:n, path; color = colors[i], linewidth = 1.2, alpha = 0.85)
    end
    return _write_png(plt, filename)
end

save_paths_png(ps::PathSet; kw...) = save_paths_png(ps.prices; kw...)

function _write_png(plt, filename::AbstractString)
    mkpath(PNG_DIR)
    out = joinpath(PNG_DIR, filename)
    try
        Plots.savefig(plt, out)
        return out
    catch err
        @warn "PNG backend failed; skipping image output" exception = err
        return nothing
    end
end

# --------------------------------------------------------------------------
# Plain-ASCII fallbacks (no dependencies at all)
# --------------------------------------------------------------------------

function _fallback_tree(io::IO, lattice::Vector{Vector{Float64}}, title::AbstractString)
    println(io, title, "  (compact view — install UnicodePlots for a real plot)")
    n = length(lattice) - 1
    for (k, row) in enumerate(lattice)
        pad = " "^(n - (k - 1))
        vals = join([@sprintf("%8.2f", v) for v in row], "  ")
        println(io, pad, vals)
    end
    return nothing
end

function _fallback_paths(io::IO, paths, title::AbstractString)
    println(io, title, "  (compact view — install UnicodePlots for a real plot)")
    all_y = reduce(vcat, paths)
    lo, hi = minimum(all_y), maximum(all_y)
    height, width = 12, 64
    canvas = fill(' ', height, width)
    for (idx, path) in enumerate(paths)
        ch = idx % 10 == 0 ? '0' : ('a' + (idx % 10))   # one glyph per path
        for (k, v) in enumerate(path)
            x = round(Int, (k - 1) / max(length(path) - 1, 1) * (width - 1)) + 1
            y = round(Int, (1 - (v - lo) / max(hi - lo, eps())) * (height - 1)) + 1
            canvas[y, x] = ch
        end
    end
    for r in 1:height
        println(io, String(canvas[r, :]))
    end
    println(io, "  prices span [", round(lo, digits = 2), ", ", round(hi, digits = 2), "]")
    return nothing
end

end # module Plotting
