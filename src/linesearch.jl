"""Working of one exact ("calculus") line search along xₖ + α pₖ."""
struct LineSearchStep
    φ::Num                    # φ(α) = F(xₖ + α pₖ)
    dφ::Num                   # φ′(α)
    roots::Vector{Float64}    # real roots of φ′(α) = 0
    α::Float64                # chosen root: minimizes φ over α > 0
end

"""
    calculus_linesearch(P::Problem)

Exact line search by calculus: build φ(α) = F(x + αp) symbolically, solve
φ′(α) = 0 (polynomial, roots via companion matrix) and take the positive root
with the smallest φ. Returns a function `(x, p) -> LineSearchStep`.
"""
function calculus_linesearch(P::Problem)
    @variables α
    return function (x, p)
        φ  = expand(substitute(P.expr, Dict(P.vars .=> x .+ α .* p)))
        dφ = expand(Symbolics.derivative(φ, α))
        rs = real_roots(dφ, α)
        φnum(a) = Symbolics.value(substitute(φ, Dict(α => a)))
        pos = filter(>(0), rs)
        isempty(pos) && error("φ′(α) = 0 has no positive root; pₖ is not a descent direction")
        return LineSearchStep(φ, dφ, rs, argmin(φnum, pos))
    end
end

"""Real roots of the univariate polynomial `poly` in `v`, ascending."""
function real_roots(poly, v; tol = 1e-9)
    n = Symbolics.degree(poly, v)
    n ≥ 1 || error("φ′(α) is constant: no stationary point")
    c = [Float64(Symbolics.value(Symbolics.coeff(poly, v^k))) for k in 1:n]
    c0 = Float64(Symbolics.value(substitute(poly, Dict(v => 0))))
    a = [c0; c] ./ c[end]                          # monic: a₀ … aₙ₋₁, 1
    n == 1 && return [-a[1]]
    C = zeros(n, n)                                # companion matrix
    C[2:end, 1:end-1] = I(n - 1)
    C[:, end] = -a[1:n]
    return sort([real(z) for z in eigvals(C) if abs(imag(z)) ≤ tol])
end
