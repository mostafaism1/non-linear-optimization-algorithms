"""One iteration of the (Fletcher–Reeves) conjugate gradient method."""
struct CGStep
    k::Int
    x::Vector{Float64}   # xₖ
    Fx::Float64          # F(xₖ)
    g::Vector{Float64}   # gₖ = ∇F(xₖ)
    β::Float64           # βₖ = ‖gₖ‖² / ‖gₖ₋₁‖²   (β₀ = 0)
    p::Vector{Float64}   # pₖ = −gₖ + βₖ pₖ₋₁
    α::Float64           # step length from the line search
    ls::LineSearchStep   # working of the line search (φ, φ′, roots)
end

"""
    conjugate_gradient(F, ∇F, x0; steps, linesearch) -> Trace{CGStep}

Fletcher–Reeves CG: p₀ = −g₀, pₖ = −gₖ + (‖gₖ‖/‖gₖ₋₁‖)² pₖ₋₁, xₖ₊₁ = xₖ + αₖ pₖ,
with αₖ from `linesearch(xₖ, pₖ, gₖ) -> LineSearchStep`. Stops early once ‖gₖ‖ ≤ `gtol`.
"""
function conjugate_gradient(F, ∇F, x0; steps, linesearch, gtol = 1e-10)
    x = float.(x0)
    records = CGStep[]
    for k in 0:steps-1
        g = ∇F(x)
        norm(g) ≤ gtol && return Trace(records, x, F(x), :converged)
        if k == 0
            β, p = 0.0, -g
        else
            prev = records[end]
            β = dot(g, g) / dot(prev.g, prev.g)
            p = -g + β * prev.p
        end
        ls = linesearch(x, p, g)
        α = ls.α
        push!(records, CGStep(k, x, F(x), g, β, p, α, ls))
        x = x + α * p
    end
    return Trace(records, x, F(x), norm(∇F(x)) ≤ gtol ? :converged : :max_steps)
end

"""
    conjugate_gradient(P::Problem, x0; steps, linesearch = calculus_linesearch(P), gtol = 1e-10)

Run CG on a symbolic `Problem`, using its derived F and ∇F.
"""
conjugate_gradient(P::Problem, x0; steps, linesearch = calculus_linesearch(P), gtol = 1e-10) =
    conjugate_gradient(P.F, P.∇F, x0; steps, linesearch, gtol)
