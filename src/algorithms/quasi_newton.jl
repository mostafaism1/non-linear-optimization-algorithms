"""Update formula that approximates the inverse Hessian: Mₖ = Aₖ ≈ H⁻¹, dₖ = −Aₖgₖ."""
abstract type InverseUpdate end
"""Update formula that approximates the Hessian: Mₖ = Bₖ ≈ H, dₖ = −Bₖ⁻¹gₖ."""
abstract type DirectUpdate end

direction(::InverseUpdate, A, g) = -A * g
direction(::DirectUpdate,  B, g) = -(B \ g)

"""
One iteration of a quasi-Newton method with update formula `U`. `M` is Aₖ ≈ H⁻¹
(`U <: InverseUpdate`) or Bₖ ≈ H (`U <: DirectUpdate`).
"""
struct QNStep{U}
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

"""
    quasi_newton(P::Problem, x0; steps, update, linesearch = calculus_linesearch(P), gtol = 1e-10)
        -> Trace{QNStep{typeof(update)}}

Quasi-Newton method with M₀ = I: dₖ = `direction(update, Mₖ, gₖ)`, xₖ₊₁ = xₖ + αₖdₖ,
then Mₖ₊₁ = `update(Mₖ, sₖ, yₖ)`. Stops early once ‖gₖ‖ ≤ `gtol`.
"""
function quasi_newton(P::Problem, x0; steps, update,
                      linesearch = calculus_linesearch(P), gtol = 1e-10)
    x = float.(x0)
    M = Matrix(1.0I, length(x), length(x))
    records = QNStep{typeof(update)}[]
    for k in 0:steps-1
        g = P.∇F(x)
        norm(g) ≤ gtol && return Trace(records, x, P.F(x), :converged)
        d = direction(update, M, g)
        ls = linesearch(x, d)
        s = ls.α * d
        y = P.∇F(x + s) - g
        M₊ = update(M, s, y)
        push!(records, QNStep{typeof(update)}(k, x, P.F(x), g, M, d, ls.α, ls, s, y, M₊))
        x, M = x + s, M₊
    end
    return Trace(records, x, P.F(x), norm(P.∇F(x)) ≤ gtol ? :converged : :max_steps)
end

"""DFP: Aₖ₊₁ = Aₖ − AₖyₖyₖᵀAₖ / (yₖᵀAₖyₖ) + sₖsₖᵀ / (sₖᵀyₖ)."""
struct DFP <: InverseUpdate end
function (::DFP)(A, s, y)
    Ay = A * y
    return A - Ay * Ay' / dot(y, Ay) + s * s' / dot(s, y)
end

"""Symmetric rank-one: Bₖ₊₁ = Bₖ + zzᵀ / (zᵀsₖ), z = yₖ − Bₖsₖ."""
struct RankOne <: DirectUpdate end
function (::RankOne)(B, s, y)
    z = y - B * s
    return B + z * z' / dot(z, s)
end

"""    dfp(P, x0; kw...) — DFP, A₀ = I (see `quasi_newton`)."""
dfp(P::Problem, x0; kw...) = quasi_newton(P, x0; update = DFP(), kw...)

"""    rank_one(P, x0; kw...) — symmetric rank-one, B₀ = I (see `quasi_newton`)."""
rank_one(P::Problem, x0; kw...) = quasi_newton(P, x0; update = RankOne(), kw...)
