"""Problem 1(i): three CG steps on F = 2x² + xy + 4y² − 14x − 19y from (0, 0), compared with the actual minimum."""
function p1i(; digits = 2)
    @variables x y
    P  = Problem(2x^2 + x*y + 4y^2 - 14x - 19y, [x, y])
    tr = conjugate_gradient(P, [0.0, 0.0]; steps = 3)
    nt = newton(P, [0.0, 0.0]; steps = 10)            # actual minimum, independent of CG
    f(v) = fmt(v; digits)
    return raw"""
    \subsection*{Problem 1(i)}
    Minimize $F(x, y) = 2x^2 + xy + 4y^2 - 14x - 19y$ from $x_0 = (0, 0)^T$ with three
    conjugate gradient (Fletcher--Reeves) steps, using calculus for each line search.

    """ * latex(tr; digits) * raw"""

    \paragraph{Remark.} $F$ is quadratic with positive definite Hessian
    $H = \begin{pmatrix} 4 & 1 \\ 1 & 8 \end{pmatrix}$, so CG with exact line searches
    converges in at most $n = 2$ steps; no third step is needed.

    \paragraph{Actual minimum.} Solving $\nabla F = 0$ gives
    """ * "\\[ x^* = $(f(nt.x)), \\quad F^* = $(f(nt.Fx)), \\]\n" *
    "which CG reached exactly in two steps.\n"
end
