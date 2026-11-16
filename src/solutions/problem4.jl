"""Problem 4: number of exact line searches until ‖xₖ₊₁ − xₖ‖ ≤ 0.1, for four methods."""
function p4(; digits = 2)
    @variables x1 x2
    P   = Problem(20x1^2 + 6x1*x2 + 5x2^2 - 62x1 - 34x2, [x1, x2])
    x0  = [0.0, 0.0]
    solve(method) = method(P, x0; steps = 50, xtol = 0.1)
    sd, cg, br = solve(steepest_descent), solve(conjugate_gradient), solve(broyden)
    nt  = newton(P, x0; steps = 50)
    ls  = calculus_linesearch(P)(x0, nt.steps[1].d)
    f(v) = fmt(v; digits)
    H, b = P.H(x0), -P.∇F(x0)
    xs   = H \ b
    n_sd, n_cg, n_nt, n_br = length(sd.steps), length(cg.steps), length(nt.steps), length(br.steps)
    last_sd = norm(sd.steps[end].α * sd.steps[end].p)
    newton_block = iteration_block(0, [
        point_lines(nt.steps[1], f)...,
        "p_{0} &= -H^{-1} g_{0} = $(f(nt.steps[1].d))",
        linesearch_lines((k = 0, ls = ls, α = ls.α), "p_{0}", f, digits)...,
        "x_{1} &= x_{0} + \\alpha_{0} p_{0} = $(f(x0 + ls.α * nt.steps[1].d)) = x^*",
    ])
    return problem_head(4, raw"""
    Given $F(x) = 20x_1^2 + 6x_1x_2 + 5x_2^2 - 62x_1 - 34x_2$ and starting at $(0, 0)$, find the optimum
    point so that $\|x_{n+1} - x_n\| \le 10^{-1}$. What is the expected number of one-dimensional
    searches required to minimize $F(x)$ using the following methods?
    \begin{itemize}
      \item[(i)] Steepest descent.
      \item[(ii)] Conjugate gradients method,
        $p_{k+1} = -g_{k+1} + \left(\frac{\|g_{k+1}\|}{\|g_k\|}\right)^2 p_k$.
      \item[(iii)] Newton's method.
      \item[(iv)] Broyden quasi-Newton method.
    \end{itemize}""") * """
    The gradient, the Hessian and the minimizer are
    \\begin{gather*}
      g = \\nabla F = (40x_1 + 6x_2 - 62,\\ 6x_1 + 10x_2 - 34)^T, \\quad H = $(f(H)), \\\\
      x^* = H^{-1} $(f(b)) = $(f(xs)).
    \\end{gather*}
    Each one-dimensional search is exact (calculus): \$\\varphi(\\alpha) = F(x_k + \\alpha p_k)\$,
    \$\\varphi'(\\alpha) = 0\$. Stop after the search that gives \$\\|x_{k+1} - x_k\\| \\le 0.1\$,
    or when \$g_{k+1} = 0\$: then \$x_{k+1} = x^*\$ and every later step is zero.

    \\paragraph{(i) Steepest descent.} \$p_k = -g_k\$.

    $(latex(sd; digits))
    Successive directions are orthogonal, so the path zig-zags to \$x^*\$. Search \$k = $(n_sd - 1)\$
    gives \$\\|x_{$n_sd} - x_{$(n_sd - 1)}\\| = $(f(last_sd)) \\le 0.1\$: \\textbf{$n_sd searches}.

    \\paragraph{(ii) Conjugate gradient.} \$p_0 = -g_0\$,
    \$p_{k+1} = -g_{k+1} + (\\|g_{k+1}\\| / \\|g_k\\|)^2 p_k\$.

    \\medskip
    $(latex_table(cg; digits))
    \$p_0 = -g_0\$, so the first search is the steepest-descent search. The directions are
    \$H\$-conjugate, so on a quadratic in \$n = 2\$ variables CG reaches \$x^*\$ after at most
    \$n\$ searches: \$g_{$n_cg} = \\nabla F(x_{$n_cg}) = $(f(cg.g))\$, \\textbf{$n_cg searches}.

    \\paragraph{(iii) Newton's method.} \$p_k = -H^{-1} g_k\$.
    $newton_block
    \$F\$ is quadratic, so the Newton direction points to \$x^*\$ and the search gives \$\\alpha_0 = 1\$.
    Then \$g_{1} = \\nabla F(x_{1}) = $(f(nt.g))\$: \\textbf{$n_nt search}.

    \\paragraph{(iv) Broyden.} \$B_0 = I\$, \$p_k = -B_k^{-1} g_k\$.

    \\medskip
    $(latex_table(br; digits))
    As in Problem 3: \$B_0 = I\$ gives the steepest-descent step. Then \$s_0^T g_1 = 0\$, so
    \$B_1 g_1 = g_1\$ and \$p_1 = -g_1\$: the second step is also the steepest-descent step
    (\$x_1\$, \$x_2\$ are the same as in (i)). Now
    \\[ B_2 = $(f(br.steps[2].M₊)) = H, \\]
    so the third step is the Newton step with \$\\alpha_2 = $(f(br.steps[3].α))\$ and \$x_3 = x^*\$: \\textbf{$n_br searches}.

    \\paragraph{Answer.}

    \\medskip
    \\begin{tabular}{l|c}
      Method & one-dimensional searches \\\\ \\hline
      (i) Steepest descent & $n_sd \\\\
      (ii) Conjugate gradient & $n_cg \\\\
      (iii) Newton & $n_nt \\\\
      (iv) Broyden & $n_br \\\\
    \\end{tabular}

    \\paragraph{Paths.} The grey contours pass through the steepest-descent points
    \$x_0, \\dots, x_{$(n_sd - 1)}\$. Each steepest-descent search stops where its line touches a
    contour, so the next direction \$-g\$ is at a right angle to the last one. CG and Broyden start
    with the same step; CG then goes to \$x^*\$, and Broyden takes a second steepest-descent step first.
    Newton goes to \$x^*\$ in one step.

    \\begin{center}
    $(latex_plot(P.F, [sd, cg, nt, br]; xlim = (-1.0, 3.3), ylim = (-0.3, 3.2),
                 levels = [s.Fx for s in sd.steps]))
    \\end{center}
    """
end
