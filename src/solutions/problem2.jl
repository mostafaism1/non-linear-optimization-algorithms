"""Problem 2: certify DFP, two DFP steps on Problem 1(i), compare with the symmetric rank-one update."""
function p2(; digits = 2)
    @variables x y
    P  = Problem(2x^2 + x*y + 4y^2 - 14x - 19y, [x, y])
    A  = dfp(P, [0.0, 0.0]; steps = 2)
    B  = rank_one(P, [0.0, 0.0]; steps = 2)
    f(v) = fmt(v; digits)
    n  = length(A.steps)
    # quasi-Newton condition on the computed numbers, one equation per step
    return problem_head(2, raw"""
    Certify that the following Davidon--Fletcher--Powell (DFP) updating formula satisfies the
    condition for a quasi-Newton method:
    \[ A_{k+1} = A_k - \frac{A_k y_k y_k^T A_k}{y_k^T A_k y_k} + \frac{s_k s_k^T}{s_k^T y_k}, \]
    where $A_k \approx H^{-1}$, $s_k = x_{k+1} - x_k$, $y_k = g_{k+1} - g_k$.
    Hence, carry out two steps to minimize $F(x)$ of Problem (1)(i) using calculus to find the minimum
    in each search direction. Compare with the updating formula given by
    \[ B_{k+1} = B_k + \frac{z z^T}{z^T s_k}, \]
    where $z = y_k - B_k s_k$, $B_k \approx H$.""") * raw"""

    \paragraph{(i) Quasi-Newton condition.} With $A_k \approx H^{-1}$, a quasi-Newton update must satisfy
    the secant condition $A_{k+1} y_k = s_k$, where $s_k = x_{k+1} - x_k$ and $y_k = g_{k+1} - g_k$
    (for a quadratic, $y_k = H s_k$, so $H^{-1} y_k = s_k$).
    \begin{proof}
    Multiply the DFP formula by $y_k$; the terms
    $y_k^T A_k y_k$ and $s_k^T y_k$ are nonzero scalars, so
    \begin{align*}
      A_{k+1} y_k &= A_k y_k - \frac{A_k y_k \,(y_k^T A_k y_k)}{y_k^T A_k y_k} + \frac{s_k \,(s_k^T y_k)}{s_k^T y_k} \\
      &= A_k y_k - A_k y_k + s_k = s_k .
    \end{align*}
    Thus DFP satisfies the quasi-Newton condition.
    \end{proof}
    $A_{k+1}$ is also symmetric when $A_k$ is.

    \paragraph{(ii) Two DFP steps on Problem 1(i).} Minimize $F(x, y) = 2x^2 + xy + 4y^2 - 14x - 19y$
    from $x_0 = (0, 0)^T$ with $A_0 = I$, using calculus for each line search. As in Problem 1(i),
    \[ g = \nabla F = (4x + y - 14,\ x + 8y - 19)^T, \quad
       H = \nabla^2 F = \begin{pmatrix} 4 & 1 \\ 1 & 8 \end{pmatrix}. \]

    """ * latex(A; digits) * """

    After \$n = $n\$ steps,
    \\[ A_{$n} = $(f(A.steps[end].M₊)) = H^{-1}. \\]

    \\paragraph{(iii) Comparison with the rank-one update.} With \$B_k \\approx H\$, the update
    \$B_{k+1} = B_k + \\frac{z z^T}{z^T s_k}\$, \$z = y_k - B_k s_k\$, satisfies the condition \$B_{k+1} s_k = y_k\$.
    \\begin{proof}
    Multiply the update by \$s_k\$; \$z^T s_k\$ is a nonzero scalar, so
    \\begin{align*}
      B_{k+1} s_k &= B_k s_k + \\frac{z\\,(z^T s_k)}{z^T s_k} = B_k s_k + z \\\\
      &= B_k s_k + y_k - B_k s_k = y_k . \\qedhere
    \\end{align*}
    \\end{proof}
    Two steps from \$x_0 = (0, 0)^T\$ with \$B_0 = I\$ and \$p_k = -B_k^{-1} g_k\$:

    """ * latex(B; digits) * """

    After \$n = $n\$ steps, \$B_{$n} = $(f(B.steps[end].M₊)) = H\$.

    \\paragraph{Comments.} Both methods reach \$x^* = $(f(A.x))\$ in \$n = $n\$ steps. DFP builds
    \$A_{$n} = H^{-1}\$, and the rank-one update builds \$B_{$n} = H\$. DFP keeps \$A_k\$ positive
    definite, but the rank-one update does not guarantee this.

    """ * captioned(latex_plot(P.F, [A, B]; xlim = (-0.3, 4.0), ylim = (-0.3, 3.0), levels = [s.Fx for s in A.steps]), """
    Paths of DFP and the rank-one update on the contours of \$F\$. On a quadratic with exact line searches and \$A_0 = B_0 = I\$, DFP and the
    rank-one update give the same points as the conjugate gradient method in Problem 1(i), so the two paths lie on one line.
    The grey contours pass through \$x_0\$ and \$x_1\$.""")
end
