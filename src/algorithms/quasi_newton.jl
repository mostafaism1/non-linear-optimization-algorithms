"""
One iteration of a quasi-Newton method. `M` is the method's matrix: Aₖ ≈ H⁻¹ for the
`:inverse` form, Bₖ ≈ H for the `:direct` form.
"""
struct QNStep
    k::Int
    x::Vector{Float64}    # xₖ
    Fx::Float64           # F(xₖ)
    g::Vector{Float64}    # gₖ = ∇F(xₖ)
    M::Matrix{Float64}    # Aₖ or Bₖ
    d::Vector{Float64}    # dₖ = −Aₖgₖ or −Bₖ⁻¹gₖ
    α::Float64            # step length from the line search
    ls::LineSearchStep    # working of the line search
    s::Vector{Float64}    # sₖ = xₖ₊₁ − xₖ
    y::Vector{Float64}    # yₖ = gₖ₊₁ − gₖ
    M₊::Matrix{Float64}   # Aₖ₊₁ or Bₖ₊₁
end

direction(::Val{:inverse}, A, g) = -A * g
direction(::Val{:direct},  B, g) = -(B \ g)

"""
    quasi_newton(P::Problem, x0; steps, form, update, linesearch = calculus_linesearch(P), gtol = 1e-10)
        -> Trace{QNStep}

Quasi-Newton method with M₀ = I: dₖ from `form` (`:inverse`: −Mₖgₖ, `:direct`: −Mₖ⁻¹gₖ),
xₖ₊₁ = xₖ + αₖdₖ, then Mₖ₊₁ = `update(Mₖ, sₖ, yₖ)`. Stops early once ‖gₖ‖ ≤ `gtol`.
"""
function quasi_newton(P::Problem, x0; steps, form, update,
                      linesearch = calculus_linesearch(P), gtol = 1e-10)
    x = float.(x0)
    M = Matrix(1.0I, length(x), length(x))
    records = QNStep[]
    for k in 0:steps-1
        g = P.∇F(x)
        norm(g) ≤ gtol && return Trace(records, x, P.F(x), :converged)
        d = direction(Val(form), M, g)
        ls = linesearch(x, d)
        s = ls.α * d
        y = P.∇F(x + s) - g
        M₊ = update(M, s, y)
        push!(records, QNStep(k, x, P.F(x), g, M, d, ls.α, ls, s, y, M₊))
        x, M = x + s, M₊
    end
    return Trace(records, x, P.F(x), norm(P.∇F(x)) ≤ gtol ? :converged : :max_steps)
end

"""DFP: Aₖ₊₁ = Aₖ − AₖyₖyₖᵀAₖ / (yₖᵀAₖyₖ) + sₖsₖᵀ / (sₖᵀyₖ)."""
function dfp_update(A, s, y)
    Ay = A * y
    return A - Ay * Ay' / dot(y, Ay) + s * s' / dot(s, y)
end

"""Symmetric rank-one: Bₖ₊₁ = Bₖ + zzᵀ / (zᵀsₖ), z = yₖ − Bₖsₖ."""
function rank_one_update(B, s, y)
    z = y - B * s
    return B + z * z' / dot(z, s)
end

"""    dfp(P, x0; kw...) — DFP, A₀ = I (see `quasi_newton`)."""
dfp(P::Problem, x0; kw...) = quasi_newton(P, x0; form = :inverse, update = dfp_update, kw...)

"""    rank_one(P, x0; kw...) — symmetric rank-one, B₀ = I (see `quasi_newton`)."""
rank_one(P::Problem, x0; kw...) = quasi_newton(P, x0; form = :direct, update = rank_one_update, kw...)
