"""Fletcher–Reeves βₖ = ‖gₖ‖² / ‖gₖ₋₁‖²."""
struct FletcherReeves end
(::FletcherReeves)(g, gprev) = dot(g, g) / dot(gprev, gprev)

"""βₖ = 0, so pₖ = −gₖ: steepest descent."""
struct SteepestDescent end
(::SteepestDescent)(g, gprev) = 0.0

"""One iteration of a conjugate-direction method with β rule `R` (e.g. `FletcherReeves`)."""
struct CGStep{R}
    k::Int
    x::Vector{Float64}   # xₖ
    Fx::Float64          # F(xₖ)
    g::Vector{Float64}   # gₖ = ∇F(xₖ)
    β::Float64           # βₖ from the rule R   (β₀ = 0)
    p::Vector{Float64}   # pₖ = −gₖ + βₖ pₖ₋₁
    α::Float64           # step length from the line search
    ls::LineSearchStep   # working of the line search (φ, φ′, roots)
end

"""
    conjugate_gradient(F, ∇F, x0; steps, linesearch) -> Trace{CGStep{typeof(β)}}

Fletcher–Reeves CG: p₀ = −g₀, pₖ = −gₖ + (‖gₖ‖/‖gₖ₋₁‖)² pₖ₋₁, xₖ₊₁ = xₖ + αₖ pₖ,
with αₖ from `linesearch(xₖ, pₖ) -> LineSearchStep`. Stops early once ‖gₖ‖ ≤ `gtol`
(`:converged`) or once a step has ‖xₖ₊₁ − xₖ‖ ≤ `xtol` (`:small_step`).
The rule `β(gₖ, gₖ₋₁)` gives βₖ; the default is `FletcherReeves()`.
"""
function conjugate_gradient(F, ∇F, x0; steps, linesearch, gtol = 1e-10, xtol = 0.0, β = FletcherReeves())
    x = float.(x0)
    records = CGStep{typeof(β)}[]
    for k in 0:steps-1
        g = ∇F(x)
        norm(g) ≤ gtol && return Trace(records, x, F(x), g, :converged)
        if k == 0
            βk, p = 0.0, -g
        else
            prev = records[end]
            βk = β(g, prev.g)
            p = -g + βk * prev.p
        end
        ls = linesearch(x, p)
        α = ls.α
        push!(records, CGStep{typeof(β)}(k, x, F(x), g, βk, p, α, ls))
        x = x + α * p
        norm(α * p) ≤ xtol && return Trace(records, x, F(x), ∇F(x), :small_step)
    end
    g = ∇F(x)
    return Trace(records, x, F(x), g, norm(g) ≤ gtol ? :converged : :max_steps)
end

"""
    conjugate_gradient(P::Problem, x0; steps, linesearch = calculus_linesearch(P), kw...)

Run CG on a symbolic `Problem`, using its derived F and ∇F.
"""
conjugate_gradient(P::Problem, x0; steps, linesearch = calculus_linesearch(P), kw...) =
    conjugate_gradient(P.F, P.∇F, x0; steps, linesearch, kw...)

"""    steepest_descent(P, x0; kw...) — pₖ = −gₖ: CG with βₖ = 0."""
steepest_descent(P::Problem, x0; kw...) = conjugate_gradient(P, x0; β = SteepestDescent(), kw...)
