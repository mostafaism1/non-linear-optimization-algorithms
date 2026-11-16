"""Problem 3: repeat Problem 2 with the Broyden update, B₀ = I, on Problem 1(i)."""
function p3(; digits = 2)
    @variables x y
    P  = Problem(2x^2 + x*y + 4y^2 - 14x - 19y, [x, y])
    B  = broyden(P, [0.0, 0.0]; steps = 2)
    C  = broyden(P, [0.0, 0.0]; steps = 3)
    f(v) = fmt(v; digits)
    n, t = length(B.steps), C.steps[end]
    check = join(("B_{$(s.k+1)} s_{$(s.k)} = $(f(s.M₊ * s.s)) = y_{$(s.k)}" for s in B.steps), ", \\quad ")
    return problem_head(3, raw"""
    Repeat (2) above using the Broyden quasi-Newton method starting from the same point, with the
    identity matrix as an initial guess for the Hessian matrix. The updating formula for the Hessian
    approximation is
    \[ B_{k+1} = B_k - \frac{(B_k s_k - y_k)\, s_k^T}{s_k^T s_k}, \quad s_k = x_{k+1} - x_k, \quad
       y_k = g_{k+1} - g_k. \]""") * raw"""

    \paragraph{(i) Quasi-Newton condition.} With $B_k \approx H$, the condition is $B_{k+1} s_k = y_k$.
    \begin{proof}
    Multiply the Broyden formula by $s_k$; $s_k^T s_k$ is a nonzero scalar, so
    \begin{align*}
      B_{k+1} s_k &= B_k s_k - \frac{(B_k s_k - y_k)\,(s_k^T s_k)}{s_k^T s_k} \\
      &= B_k s_k - B_k s_k + y_k = y_k .
    \end{align*}
    Thus the Broyden update satisfies the quasi-Newton condition.
    \end{proof}
    The Broyden update does not keep $B_k$ symmetric.

    \paragraph{(ii) Two Broyden steps on Problem 1(i).} Minimize $F(x, y) = 2x^2 + xy + 4y^2 - 14x - 19y$
    from $x_0 = (0, 0)^T$ with $B_0 = I$ and $p_k = -B_k^{-1} g_k$, using calculus for each line search.
    As in Problem 1(i),
    \[ g = \nabla F = (4x + y - 14,\ x + 8y - 19)^T, \quad
       H = \nabla^2 F = \begin{pmatrix} 4 & 1 \\ 1 & 8 \end{pmatrix}. \]

    """ * latex(B; digits) * """

    \\paragraph{Check.} On the computed numbers:
    \\[ $check. \\]
    After \$n = $n\$ steps, \$B_{$n} = $(f(B.steps[end].M₊)) = H\$, but
    \$x_{$n} = $(f(B.x)) \\neq x^*\$, because \$g_{$n} = $(f(P.∇F(B.x))) \\neq 0\$.

    \\paragraph{Third step.} In iteration \$k=1\$, \$p_{1} = -g_{1}\$: the second step is also a
    steepest-descent step, so two steps do not reach \$x^*\$. But \$B_{$n} = H\$, so the next step is
    the Newton step:
    \\[ p_{2} = -B_{2}^{-1} g_{2} = $(f(t.p)), \\quad \\alpha_{2} = $(f(t.α)), \\quad
       x_{3} = $(f(C.x)) = x^* . \\]

    \\paragraph{(iii) Comments.} DFP and the rank-one update reach \$x^*\$ in \$n = 2\$ steps. Broyden
    needs $(length(C.steps)), because its second step is also a steepest-descent step. Also, DFP and the
    rank-one update keep the matrix symmetric, but Broyden does not (see \$B_1\$).

    """ * captioned(latex_plot(P.F, [dfp(P, [0.0, 0.0]; steps = 2), C]; xlim = (-0.3, 4.0), ylim = (-0.3, 3.0),
                 levels = [s.Fx for s in C.steps]), """
    Paths of DFP and Broyden on the contours of \$F\$. DFP (Problem 2) and Broyden take the same first step. DFP then goes to \$x^*\$.
    Broyden takes a second steepest-descent step to \$x_2\$, then the Newton step to \$x^*\$.
    The grey contours pass through the Broyden points \$x_0\$, \$x_1\$, \$x_2\$.""")
end
