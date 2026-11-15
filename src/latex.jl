using Printf: Format, format

"""
    fmt(x::Real; digits = 2) -> String

Fixed-point with `digits` decimals. A value that rounds to zero prints unsigned (never "-0.00").
"""
function fmt(x::Real; digits = 2)
    s = format(Format("%.$(digits)f"), x)
    return startswith(s, '-') && all(c -> c in "-0.", s) ? s[2:end] : s
end

"""    fmt(v::AbstractVector; digits = 2) -> String — column vector as `(v₁,\\ v₂)^T`."""
fmt(v::AbstractVector; digits = 2) = "(" * join((fmt(x; digits) for x in v), ",\\ ") * ")^T"

"""    fmt(M::AbstractMatrix; digits = 2) -> String — matrix as `pmatrix`."""
fmt(M::AbstractMatrix; digits = 2) =
    "\\begin{pmatrix} " *
    join((join((fmt(x; digits) for x in r), " & ") for r in eachrow(M)), " \\\\ ") *
    " \\end{pmatrix}"

"""
    fmt_poly(c; digits = 2) -> String

Polynomial `c[1] + c[2] α + … + c[n+1] αⁿ`, written in descending powers. Terms that
round to zero at `digits` are dropped.
"""
function fmt_poly(c::AbstractVector; digits = 2)
    mono(k) = k == 0 ? "" : k == 1 ? "\\,\\alpha" : "\\,\\alpha^{$k}"
    terms = [(c[k+1], k) for k in length(c)-1:-1:0 if fmt(abs(c[k+1]); digits) != fmt(0; digits)]
    isempty(terms) && return fmt(0; digits)
    out = ""
    for (i, (ck, k)) in enumerate(terms)
        sign = ck < 0 ? (i == 1 ? "-" : " - ") : (i == 1 ? "" : " + ")
        out *= sign * fmt(abs(ck); digits) * mono(k)
    end
    return out
end

"""Lines for xₖ, F(xₖ) and gₖ, common to every line-search method."""
point_lines(s, f) = ["x_{$(s.k)} &= $(f(s.x)), \\quad F(x_{$(s.k)}) = $(f(s.Fx))",
                     "g_{$(s.k)} &= \\nabla F(x_{$(s.k)}) = $(f(s.g))"]

"""Lines for the calculus line search along direction `dir` (e.g. "p_{0}"): φ(α), φ′(α) = 0 ⇒ αₖ."""
linesearch_lines(s, dir, f, digits) = [
    "\\varphi(\\alpha) &= F(x_{$(s.k)} + \\alpha $dir) = $(fmt_poly(poly_coeffs(s.ls.φ); digits))",
    "\\varphi'(\\alpha) &= $(fmt_poly(poly_coeffs(s.ls.dφ); digits)) = 0 \\;\\Rightarrow\\; \\alpha_{$(s.k)} = $(f(s.α))"]

"""An iteration's lines as a `\\paragraph` heading plus an `align*` block."""
iteration_block(k, lines) = "\\paragraph{Iteration \$k=$k\$}\n\\begin{align*}\n" *
                            join(("  " * l for l in lines), "\\\\\n") * "\n\\end{align*}\n"

"""
    latex(s::CGStep; digits = 2) -> String

Derivation block for one CG iteration: x, F, g, p, φ(α), φ′(α) = 0 ⇒ α, and the next x.
"""
function latex(s::CGStep; digits = 2)
    k, f = s.k, v -> fmt(v; digits)
    xk, gk, pk, αk = "x_{$k}", "g_{$k}", "p_{$k}", "\\alpha_{$k}"
    return iteration_block(k, [
        point_lines(s, f)...,
        (k == 0 ? ["$pk &= -$gk = $(f(s.p))"] :
         ["\\beta_{$k} &= \\frac{\\|$gk\\|^2}{\\|g_{$(k-1)}\\|^2} = $(f(s.β))",
          "$pk &= -$gk + \\beta_{$k} p_{$(k-1)} = $(f(s.p))"])...,
        linesearch_lines(s, pk, f, digits)...,
        "x_{$(k+1)} &= $xk + $αk $pk = $(f(s.x + s.α * s.p))",
    ])
end

"""`tabular` with a `c|c…c` layout: header row, `\\hline`, then one line per row of cells."""
function tabular(header, rows)
    row(cells) = "  " * join(cells, " & ") * " \\\\"
    return "\\begin{tabular}{c|$("c"^(length(header) - 1))}\n" *
           join([row(header) * " \\hline"; row.(rows)], "\n") * "\n\\end{tabular}\n"
end

"""
    latex(tr::Trace; digits = 2) -> String

All iteration blocks, then (if the run converged) the stop sentence, then the summary table.
"""
function latex(tr::Trace; digits = 2)
    n = length(tr.steps)
    stop = tr.status == :converged ?
        "\$g_{$n} = 0\$, so the method stops at \$x_{$n} = x^*\$.\n\n" : "\n"
    return join((latex(s; digits) for s in tr.steps), "\n") * "\n" * stop *
           "\\medskip\n" * latex_table(tr; digits)
end

direction_tex(::InverseUpdate, k) = "-A_{$k} g_{$k}"
direction_tex(::DirectUpdate,  k) = "-B_{$k}^{-1} g_{$k}"

"""Lines that show the matrix update Mₖ → Mₖ₊₁ for update formula `U`."""
function update_lines(::DFP, s, f)
    k, j = s.k, s.k + 1
    return ["A_{$j} &= A_{$k} - \\frac{A_{$k} y_{$k} y_{$k}^T A_{$k}}{y_{$k}^T A_{$k} y_{$k}} + " *
            "\\frac{s_{$k} s_{$k}^T}{s_{$k}^T y_{$k}} = $(f(s.M₊))"]
end

function update_lines(::RankOne, s, f)
    k, j = s.k, s.k + 1
    return ["z_{$k} &= y_{$k} - B_{$k} s_{$k} = $(f(s.y - s.M * s.s))",
            "B_{$j} &= B_{$k} + \\frac{z_{$k} z_{$k}^T}{z_{$k}^T s_{$k}} = $(f(s.M₊))"]
end

function update_lines(::Broyden, s, f)
    k, j = s.k, s.k + 1
    return ["B_{$k} s_{$k} - y_{$k} &= $(f(s.M * s.s - s.y))",
            "B_{$j} &= B_{$k} - \\frac{(B_{$k} s_{$k} - y_{$k}) s_{$k}^T}{s_{$k}^T s_{$k}} = $(f(s.M₊))"]
end

"""
    latex(s::QNStep; digits = 2) -> String

Derivation block for one quasi-Newton iteration: x, F, g, d, φ(α), φ′(α) = 0 ⇒ α, s and the
next x, y, then the matrix update of the method.
"""
function latex(s::QNStep{U}; digits = 2) where {U}
    k, f = s.k, v -> fmt(v; digits)
    xk, gk, pk, αk = "x_{$k}", "g_{$k}", "p_{$k}", "\\alpha_{$k}"
    return iteration_block(k, [
        point_lines(s, f)...,
        "$pk &= $(direction_tex(U(), k)) = $(f(s.d))",
        linesearch_lines(s, pk, f, digits)...,
        "s_{$k} &= $αk $pk = $(f(s.s)), \\quad x_{$(k+1)} = $xk + s_{$k} = $(f(s.x + s.s))",
        "y_{$k} &= g_{$(k+1)} - $gk = $(f(s.y))",
        update_lines(U(), s, f)...,
    ])
end

search_dir(s::CGStep) = s.p
search_dir(s::QNStep) = s.d

"""Columns that only some methods have: βₖ for Fletcher–Reeves CG (β₀ shown as "---")."""
extra_header(::Type)                            = String[]
extra_header(::Type{CGStep{FletcherReeves}})    = ["\$\\beta_k\$"]
extra_cells(s, m)                               = String[]
extra_cells(s::CGStep{FletcherReeves}, m)       = [s.k == 0 ? "---" : m(s.β)]

"""
    latex_table(tr::Trace{S}; digits = 2) -> String

Summary table: one row per line search (k, xₖ, F(xₖ), the method's extra columns, pₖ, αₖ,
‖xₖ₊₁ − xₖ‖) plus a final row for the end point. αₖ gets two more decimals than the rest,
since short steps round to almost nothing.
"""
function latex_table(tr::Trace{S}; digits = 2) where {S}
    m(v; d = digits) = "\$" * fmt(v; digits = d) * "\$"
    extra = extra_header(S)
    return tabular(["\$k\$", "\$x_k\$", "\$F(x_k)\$", extra..., "\$p_k\$", "\$\\alpha_k\$",
                    "\$\\|x_{k+1} - x_k\\|\$"],
                   [[[s.k, m(s.x), m(s.Fx), extra_cells(s, m)..., m(search_dir(s)),
                      m(s.α; d = digits + 2), m(norm(s.α * search_dir(s)))] for s in tr.steps];
                    [[length(tr.steps), m(tr.x), m(tr.Fx), fill("---", length(extra) + 3)...]]])
end
