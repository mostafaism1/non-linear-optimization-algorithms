"""
    Trace{S}

Per-iteration record of an algorithm run: `steps` holds one record of type `S`
per iteration, `x` is the final iterate with objective value `Fx` and gradient `g`, `status` is why the run stopped
(`:converged` — gradient vanished; `:max_steps` — iteration budget used up;
`:small_step` — last step ‖xₖ₊₁ − xₖ‖ ≤ `xtol`).
"""
struct Trace{S}
    steps::Vector{S}
    x::Vector{Float64}
    Fx::Float64
    g::Vector{Float64}
    status::Symbol
end
