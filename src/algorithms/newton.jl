"""One Newton iteration: at `x` (objective `Fx`), gradient `g`, Hessian `H`, step `d = −H⁻¹g`."""
struct NewtonStep
    k::Int
    x::Vector{Float64}
    Fx::Float64
    g::Vector{Float64}
    H::Matrix{Float64}
    d::Vector{Float64}
end

"""
    newton(P::Problem, x0; steps, gtol = 1e-10) -> Trace{NewtonStep}

Pure Newton iteration xₖ₊₁ = xₖ − H(xₖ)⁻¹∇F(xₖ). Stops early once ‖gₖ‖ ≤ `gtol`.
"""
function newton(P::Problem, x0; steps, gtol = 1e-10)
    x = float.(x0)
    records = NewtonStep[]
    for k in 0:steps-1
        g = P.∇F(x)
        norm(g) ≤ gtol && return Trace(records, x, P.F(x), g, :converged)
        H = P.H(x)
        d = -(H \ g)
        push!(records, NewtonStep(k, x, P.F(x), g, H, d))
        x = x + d
    end
    g = P.∇F(x)
    return Trace(records, x, P.F(x), g, norm(g) ≤ gtol ? :converged : :max_steps)
end
