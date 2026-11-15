"""
    Problem(expr, vars)

An objective given symbolically. Keeps the symbolic F, ∇F and Hessian (for the
LaTeX printer and the calculus line search) and compiles numeric callables
`F(x)`, `∇F(x)`, `H(x)` for the algorithms.
"""
struct Problem{TF, TG, TH}
    vars::Vector{Num}
    expr::Num
    ∇expr::Vector{Num}
    Hexpr::Matrix{Num}
    F::TF
    ∇F::TG
    H::TH
end

function Problem(expr, vars)
    vars = collect(Num, vars)
    ∇expr = Symbolics.gradient(expr, vars)
    Hexpr = Symbolics.hessian(expr, vars)
    compile(e) = build_function(e, vars; expression = Val(false))
    F = compile(expr)
    ∇F = first(compile(∇expr))      # out-of-place variant
    H = first(compile(Hexpr))
    return Problem(vars, Num(expr), ∇expr, Hexpr, F, ∇F, H)
end
