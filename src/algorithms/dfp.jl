"""One iteration of the Davidon–Fletcher–Powell quasi-Newton method (Aₖ ≈ H⁻¹)."""
struct DFPStep
    k::Int
    x::Vector{Float64}    # xₖ
    Fx::Float64           # F(xₖ)
    g::Vector{Float64}    # gₖ = ∇F(xₖ)
    A::Matrix{Float64}    # Aₖ
    d::Vector{Float64}    # dₖ = −Aₖ gₖ
    α::Float64            # step length from the line search
    ls::LineSearchStep    # working of the line search
    s::Vector{Float64}    # sₖ = xₖ₊₁ − xₖ
    y::Vector{Float64}    # yₖ = gₖ₊₁ − gₖ
    A₊::Matrix{Float64}   # Aₖ₊₁ (DFP update)
end

"""
    dfp(P::Problem, x0; steps, linesearch = calculus_linesearch(P), gtol = 1e-10) -> Trace{DFPStep}

DFP with A₀ = I: dₖ = −Aₖgₖ, xₖ₊₁ = xₖ + αₖdₖ, then
Aₖ₊₁ = Aₖ − AₖyₖyₖᵀAₖ / (yₖᵀAₖyₖ) + sₖsₖᵀ / (sₖᵀyₖ). Stops early once ‖gₖ‖ ≤ `gtol`.
"""
function dfp(P::Problem, x0; steps, linesearch = calculus_linesearch(P), gtol = 1e-10)
    x = float.(x0)
    A = Matrix(1.0I, length(x), length(x))
    records = DFPStep[]
    for k in 0:steps-1
        g = P.∇F(x)
        norm(g) ≤ gtol && return Trace(records, x, P.F(x), :converged)
        d = -A * g
        ls = linesearch(x, d)
        s = ls.α * d
        y = P.∇F(x + s) - g
        Ay = A * y
        A₊ = A - Ay * Ay' / dot(y, Ay) + s * s' / dot(s, y)
        push!(records, DFPStep(k, x, P.F(x), g, A, d, ls.α, ls, s, y, A₊))
        x, A = x + s, A₊
    end
    return Trace(records, x, P.F(x), norm(P.∇F(x)) ≤ gtol ? :converged : :max_steps)
end
