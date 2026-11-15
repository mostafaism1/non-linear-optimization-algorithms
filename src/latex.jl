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

"""
    latex(s::CGStep; digits = 2) -> String

Derivation block for one CG iteration: x, F, g, p, φ(α), φ′(α) = 0 ⇒ α, and the next x.
"""
function latex(s::CGStep; digits = 2)
    k, f = s.k, v -> fmt(v; digits)
    xk, gk, pk, αk = "x_{$k}", "g_{$k}", "p_{$k}", "\\alpha_{$k}"
    lines = [
        "$xk &= $(f(s.x)), \\quad F($xk) = $(f(s.Fx))",
        "$gk &= \\nabla F($xk) = $(f(s.g))",
        (k == 0 ? ["$pk &= -$gk = $(f(s.p))"] :
         ["\\beta_{$k} &= \\frac{\\|$gk\\|^2}{\\|g_{$(k-1)}\\|^2} = $(f(s.β))",
          "$pk &= -$gk + \\beta_{$k} p_{$(k-1)} = $(f(s.p))"])...,
        "\\varphi(\\alpha) &= F($xk + \\alpha $pk) = $(fmt_poly(poly_coeffs(s.ls.φ); digits))",
        "\\varphi'(\\alpha) &= $(fmt_poly(poly_coeffs(s.ls.dφ); digits)) = 0 \\;\\Rightarrow\\; $αk = $(f(s.α))",
        "x_{$(k+1)} &= $xk + $αk $pk = $(f(s.x + s.α * s.p))",
    ]
    return "\\paragraph{Iteration \$k=$k\$}\n\\begin{align*}\n" *
           join(("  " * l for l in lines), "\\\\\n") * "\n\\end{align*}\n"
end

"""
    latex_table(tr::Trace{CGStep}; digits = 2) -> String

Summary table: one row per iteration (k, xₖ, F(xₖ), βₖ, αₖ) plus a final row for the end point.
β₀ and the end point's β, α are shown as "---".
"""
function latex_table(tr::Trace{CGStep}; digits = 2)
    m(v) = "\$" * fmt(v; digits) * "\$"
    row(cells...) = "  " * join(cells, " & ") * " \\\\"
    rows = [row(s.k, m(s.x), m(s.Fx), s.k == 0 ? "---" : m(s.β), m(s.α)) for s in tr.steps]
    push!(rows, row(length(tr.steps), m(tr.x), m(tr.Fx), "---", "---"))
    header = row("\$k\$", "\$x_k\$", "\$F(x_k)\$", "\$\\beta_k\$", "\$\\alpha_k\$") * " \\hline"
    return "\\begin{tabular}{c|cccc}\n" * join([header; rows], "\n") * "\n\\end{tabular}\n"
end

"""
    latex(tr::Trace{CGStep}; digits = 2) -> String

All iteration blocks, then (if the run converged) the stop sentence, then the summary table.
"""
function latex(tr::Trace{CGStep}; digits = 2)
    n = length(tr.steps)
    stop = tr.status == :converged ?
        "\$g_{$n} = 0\$, so the method stops at \$x_{$n} = x^*\$.\n\n" : "\n"
    return join((latex(s; digits) for s in tr.steps), "\n") * "\n" * stop *
           "\\medskip\n" * latex_table(tr; digits)
end
