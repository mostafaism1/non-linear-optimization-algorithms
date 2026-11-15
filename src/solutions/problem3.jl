"""Problem 3: repeat Problem 2 with the Broyden update, B₀ = I, on Problem 1(i)."""
function p3(; digits = 2)
    @variables x y
    P  = Problem(2x^2 + x*y + 4y^2 - 14x - 19y, [x, y])
    B  = broyden(P, [0.0, 0.0]; steps = 2)
    C  = broyden(P, [0.0, 0.0]; steps = 3)
    f(v) = fmt(v; digits)
    n, t = length(B.steps), C.steps[end]
    check = join(("B_{$(s.k+1)} s_{$(s.k)} = $(f(s.M₊ * s.s)) = y_{$(s.k)}" for s in B.steps), ", \\quad ")
    return raw"""
    \section*{Problem 3}

    \paragraph{(a) Quasi-Newton condition.} With $B_k \approx H$, the condition is $B_{k+1} s_k = y_k$.
    Multiply the Broyden formula by $s_k$; $s_k^T s_k$ is a nonzero scalar, so
    \begin{align*}
      B_{k+1} s_k &= B_k s_k - \frac{(B_k s_k - y_k)\,(s_k^T s_k)}{s_k^T s_k} \\
      &= B_k s_k - B_k s_k + y_k = y_k .
    \end{align*}
    Thus the Broyden update satisfies the quasi-Newton condition. It does not keep $B_k$ symmetric.

    \paragraph{(b) Two Broyden steps on Problem 1(i).} Minimize $F(x, y) = 2x^2 + xy + 4y^2 - 14x - 19y$
    from $x_0 = (0, 0)^T$ with $B_0 = I$ and $d_k = -B_k^{-1} g_k$, using calculus for each line search.

    """ * latex(B; digits) * """

    \\paragraph{Check.} On the computed numbers:
    \\[ $check. \\]
    After \$n = $n\$ steps, \$B_{$n} = $(f(B.steps[end].M₊)) = H\$, but
    \$x_{$n} = $(f(B.x)) \\neq x^*\$, because \$g_{$n} = $(f(P.∇F(B.x))) \\neq 0\$.

    \\paragraph{Third step.} In iteration \$k=1\$, \$d_{1} = -g_{1}\$: the second step is also a
    steepest-descent step, so two steps do not reach \$x^*\$. But \$B_{$n} = H\$, so the next step is
    the Newton step:
    \\[ d_{2} = -B_{2}^{-1} g_{2} = $(f(t.d)), \\quad \\alpha_{2} = $(f(t.α)), \\quad
       x_{3} = $(f(C.x)) = x^* . \\]

    \\paragraph{Comparison with Problem 2.}
    \\begin{itemize}
      \\item All three updates satisfy the quasi-Newton condition and start from the identity,
            so the first step is \$d_0 = -g_0\$.
      \\item DFP and the rank-one update reach \$x^*\$ in \$n = 2\$ steps. Broyden needs $(length(C.steps)) steps on this
            problem: its second step is also a steepest-descent step.
      \\item DFP and the rank-one update keep the matrix symmetric. Broyden does not
            (here \$B_1 = $(f(B.steps[1].M₊))\$).
      \\item Broyden needs only \$s_k^T s_k \\neq 0\$, which holds for every nonzero step.
    \\end{itemize}
    """
end
