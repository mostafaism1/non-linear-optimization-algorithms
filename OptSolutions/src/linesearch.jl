"""
    quadratic_linesearch(H)

Exact line search for a quadratic with Hessian `H`: α = −gᵀp / pᵀHp.
Returns a function `(x, p, g) -> α`.
"""
quadratic_linesearch(H) = (x, p, g) -> -dot(g, p) / dot(p, H * p)
