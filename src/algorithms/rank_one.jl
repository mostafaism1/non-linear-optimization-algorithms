"""One iteration of the symmetric rank-one quasi-Newton method (Bₖ ≈ H)."""
struct RankOneStep
    k::Int
    x::Vector{Float64}    # xₖ
    Fx::Float64           # F(xₖ)
    g::Vector{Float64}    # gₖ = ∇F(xₖ)
    B::Matrix{Float64}    # Bₖ
    d::Vector{Float64}    # dₖ = −Bₖ⁻¹ gₖ
    α::Float64            # step length from the line search
    ls::LineSearchStep    # working of the line search
    s::Vector{Float64}    # sₖ = xₖ₊₁ − xₖ
    y::Vector{Float64}    # yₖ = gₖ₊₁ − gₖ
    z::Vector{Float64}    # z = yₖ − Bₖ sₖ
    B₊::Matrix{Float64}   # Bₖ₊₁ (rank-one update)
end

"""
    rank_one(P::Problem, x0; steps, linesearch = calculus_linesearch(P), gtol = 1e-10) -> Trace{RankOneStep}

Symmetric rank-one method with B₀ = I: dₖ = −Bₖ⁻¹gₖ, xₖ₊₁ = xₖ + αₖdₖ, then
Bₖ₊₁ = Bₖ + zzᵀ / (zᵀsₖ) with z = yₖ − Bₖsₖ. Stops early once ‖gₖ‖ ≤ `gtol`.
"""
function rank_one(P::Problem, x0; steps, linesearch = calculus_linesearch(P), gtol = 1e-10)
    x = float.(x0)
    B = Matrix(1.0I, length(x), length(x))
    records = RankOneStep[]
    for k in 0:steps-1
        g = P.∇F(x)
        norm(g) ≤ gtol && return Trace(records, x, P.F(x), :converged)
        d = -(B \ g)
        ls = linesearch(x, d)
        s = ls.α * d
        y = P.∇F(x + s) - g
        z = y - B * s
        B₊ = B + z * z' / dot(z, s)
        push!(records, RankOneStep(k, x, P.F(x), g, B, d, ls.α, ls, s, y, z, B₊))
        x, B = x + s, B₊
    end
    return Trace(records, x, P.F(x), norm(P.∇F(x)) ≤ gtol ? :converged : :max_steps)
end
