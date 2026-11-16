"""Problem 1(i): three CG steps on F = 2x² + xy + 4y² − 14x − 19y from (0, 0), compared with the actual minimum."""
function p1i(; digits = 2)
    @variables x y
    P  = Problem(2x^2 + x*y + 4y^2 - 14x - 19y, [x, y])
    tr = conjugate_gradient(P, [0.0, 0.0]; steps = 3)
    nt = newton(P, [0.0, 0.0]; steps = 10)            # actual minimum, independent of CG
    f(v) = fmt(v; digits)
    return problem_head(1, raw"""
    Carry out three steps of the conjugate gradient method to find the minimum point of the functions
    \begin{gather*}
      \text{(i)}\quad F(x, y) = 2x^2 + xy + 4y^2 - 14x - 19y, \\
      \text{(ii)}\quad F(x) = 2x_1^2 x_2^2 + (x_1 - 2)^2 + (x_2 - 1)^2.
    \end{gather*}
    Start at $(0, 0)$ and use calculus to find the minimum in each conjugate direction.
    Comment on your result after the three steps in comparison with the actual minimum point.""") * raw"""
    \subsection*{(i) $F(x, y) = 2x^2 + xy + 4y^2 - 14x - 19y$}
    \begin{align*}
      F_x &= 4x + y - 14 \\
      F_y &= x + 8y - 19 \\
      g = \nabla F &= (4x + y - 14,\ x + 8y - 19)^T
    \end{align*}
    Three Fletcher--Reeves steps from $x_0 = (0, 0)^T$.

    """ * latex(tr; digits) * raw"""

    \paragraph{Actual minimum.} Solving $\nabla F = 0$ gives
    """ * "\\[ x^* = $(f(nt.x)), \\quad F^* = $(f(nt.Fx)). \\]\n" * raw"""

    \paragraph{Comments.} $F$ is quadratic with positive definite Hessian
    $H = \begin{pmatrix} 4 & 1 \\ 1 & 8 \end{pmatrix}$, so CG with exact line searches
    converges in at most $n = 2$ steps. CG reached $x^*$ exactly in two steps; the third direction is
    $p_2 = 0$, so the third step does not move.
    """ * """

    \\paragraph{Path.} The grey contours pass through \$x_0\$ and \$x_1\$. The first search stops where
    its line touches the contour through \$x_1\$. The second direction is \$H\$-conjugate to the first
    and goes to \$x^*\$.

    \\begin{center}
    $(latex_plot(P.F, [tr]; xlim = (-0.3, 4.0), ylim = (-0.3, 3.0), levels = [s.Fx for s in tr.steps]))
    \\end{center}
    """
end

"""Problem 1(ii): three CG steps on F = 2x₁²x₂² + (x₁−2)² + (x₂−1)² from (0, 0), compared with the actual minimum."""
function p1ii(; digits = 2, cmp_digits = 4)
    @variables x1 x2
    P  = Problem(2x1^2 * x2^2 + (x1 - 2)^2 + (x2 - 1)^2, [x1, x2])
    tr = conjugate_gradient(P, [0.0, 0.0]; steps = 3)
    nt = newton(P, tr.x; steps = 20)                  # actual minimum, from the CG endpoint
    λ  = eigvals(Symmetric(P.H(nt.x)))
    f(v) = fmt(v; digits)
    g(v) = fmt(v; digits = cmp_digits)
    n  = length(tr.steps)
    return raw"""
    \subsection*{(ii) $F(x_1, x_2) = 2x_1^2 x_2^2 + (x_1 - 2)^2 + (x_2 - 1)^2$}
    \begin{align*}
      F_{x_1} &= 4x_1 x_2^2 + 2(x_1 - 2) \\
      F_{x_2} &= 4x_1^2 x_2 + 2(x_2 - 1) \\
      g = \nabla F &= (4x_1 x_2^2 + 2(x_1 - 2),\ 4x_1^2 x_2 + 2(x_2 - 1))^T
    \end{align*}
    Three Fletcher--Reeves steps from $x_0 = (0, 0)^T$.

    """ * latex(tr; digits) * """

    \\paragraph{Actual minimum.} Newton's method on \$\\nabla F = 0\$, started at \$x_{$n}\$, converges in $(length(nt.steps)) iterations to
    \\[ x^* = $(f(nt.x)), \\quad F^* = $(f(nt.Fx)), \\]
    where \$H(x^*)\$ is positive definite (eigenvalues $(f(λ[1])) and $(f(λ[2]))), so \$x^*\$ is a strict local minimum.

    \\paragraph{Comments.} \$F\$ is not quadratic, so CG has no finite-termination guarantee.
    After three steps (to $cmp_digits decimal places):
    \\[ F(x_{$n}) = $(g(tr.Fx)), \\quad F^* = $(g(nt.Fx)), \\quad \\|x_{$n} - x^*\\| = $(g(norm(tr.x - nt.x))). \\]

    \\paragraph{Path.} The grey contours pass through \$x_0\$, \$x_1\$ and \$x_2\$. \$F\$ is not quadratic,
    so the contours are not ellipses. Each search stops where its line touches the contour through
    the next point. The last point is \$x_{$n}\$; \$x^*\$ is only $(g(norm(tr.x - nt.x))) from it, so
    in the figure the two points are at the same position.

    \\begin{center}
    $(latex_plot(P.F, [tr]; xlim = (-0.3, 2.3), ylim = (-0.3, 1.0), levels = [s.Fx for s in tr.steps]))
    \\end{center}
    """
end
