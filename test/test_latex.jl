@testset "latex formatting" begin
    @testset "numbers: fixed 2 d.p. by default" begin
        @test fmt(0.132493) == "0.13"
        @test fmt(-14) == "-14.00"
        @test fmt(-0.001) == "0.00"                # no "-0.00"
        @test fmt(0.88634; digits = 4) == "0.8863" # extra precision on demand
    end

    @testset "vectors: column vector as transposed tuple" begin
        @test fmt([1.8549, 2.5174]) == raw"(1.85,\ 2.52)^T"
        @test fmt([-14.0, -19.0]; digits = 1) == raw"(-14.0,\ -19.0)^T"
    end
end

@testset "polynomials in α: descending powers, 2 d.p., zero terms dropped" begin
    @test fmt_poly([0.0, -557.0, 2102.0]) == raw"2102.00\,\alpha^{2} - 557.00\,\alpha"
    @test fmt_poly([-557.0, 4204.0]) == raw"4204.00\,\alpha - 557.00"
    @test fmt_poly([5.0, 0.0, -1.0]) == raw"-1.00\,\alpha^{2} + 5.00"
    @test fmt_poly([0.001, 1.0]; digits = 2) == raw"1.00\,\alpha"   # rounds to zero ⇒ dropped
end

@testset "CG iteration block (1(i), k = 0)" begin
    tr = conjugate_gradient(P1i, [0.0, 0.0]; steps = 3)
    @test latex(tr.steps[1]) == raw"""
    \paragraph{Iteration $k=0$}
    \begin{align*}
      x_{0} &= (0.00,\ 0.00)^T, \quad F(x_{0}) = 0.00\\
      g_{0} &= \nabla F(x_{0}) = (-14.00,\ -19.00)^T\\
      p_{0} &= -g_{0} = (14.00,\ 19.00)^T\\
      \varphi(\alpha) &= F(x_{0} + \alpha p_{0}) = 2102.00\,\alpha^{2} - 557.00\,\alpha\\
      \varphi'(\alpha) &= 4204.00\,\alpha - 557.00 = 0 \;\Rightarrow\; \alpha_{0} = 0.13\\
      x_{1} &= x_{0} + \alpha_{0} p_{0} = (1.85,\ 2.52)^T
    \end{align*}
    """
end

@testset "CG iteration block, k ≥ 1: Fletcher–Reeves β and p" begin
    tr = conjugate_gradient(P1i, [0.0, 0.0]; steps = 3)
    s1 = latex(tr.steps[2])
    # β₁ = ‖g₁‖²/‖g₀‖² ≈ 25.4710/557 ≈ 0.0457;  p₁ = −g₁ + β₁p₀ ≈ (4.70, −2.12)
    @test occursin(raw"\beta_{1} &= \frac{\|g_{1}\|^2}{\|g_{0}\|^2} = 0.05\\", s1)
    @test occursin(raw"p_{1} &= -g_{1} + \beta_{1} p_{0} = (4.70,\ -2.12)^T\\", s1)
    @test !occursin("p_{1} &= -g_{1} =", s1)
    @test !occursin(raw"\beta", latex(tr.steps[1]))       # k = 0: no β line
end

@testset "CG summary table (1(i))" begin
    tr = conjugate_gradient(P1i, [0.0, 0.0]; steps = 3)
    @test latex_table(tr) == raw"""
    \begin{tabular}{c|cccc}
      $k$ & $x_k$ & $F(x_k)$ & $\beta_k$ & $\alpha_k$ \\ \hline
      0 & $(0.00,\ 0.00)^T$ & $0.00$ & --- & $0.13$ \\
      1 & $(1.85,\ 2.52)^T$ & $-36.90$ & $0.05$ & $0.24$ \\
      2 & $(3.00,\ 2.00)^T$ & $-40.00$ & --- & --- \\
    \end{tabular}
    """
end

@testset "whole CG trace: blocks, stop sentence, table" begin
    tr = conjugate_gradient(P1i, [0.0, 0.0]; steps = 3)
    out = latex(tr)
    @test out == join(latex.(tr.steps), "\n") * "\n" *
                 raw"$g_{2} = 0$, so the method stops at $x_{2} = x^*$." * "\n\n" *
                 raw"\medskip" * "\n" * latex_table(tr)          # space before the table

    short = conjugate_gradient(P1i, [0.0, 0.0]; steps = 1)   # :max_steps ⇒ no stop sentence
    @test !occursin("stops", latex(short))
end

@testset "matrices: pmatrix, 2 d.p." begin
    @test fmt([1.0 0.0; 0.0 1.0]) == raw"\begin{pmatrix} 1.00 & 0.00 \\ 0.00 & 1.00 \end{pmatrix}"
end

@testset "DFP iteration block (1(i), k = 0)" begin
    tr = dfp(P1i, [0.0, 0.0]; steps = 2)
    @test latex(tr.steps[1]) == raw"""
    \paragraph{Iteration $k=0$}
    \begin{align*}
      x_{0} &= (0.00,\ 0.00)^T, \quad F(x_{0}) = 0.00\\
      g_{0} &= \nabla F(x_{0}) = (-14.00,\ -19.00)^T\\
      d_{0} &= -A_{0} g_{0} = (14.00,\ 19.00)^T\\
      \varphi(\alpha) &= F(x_{0} + \alpha d_{0}) = 2102.00\,\alpha^{2} - 557.00\,\alpha\\
      \varphi'(\alpha) &= 4204.00\,\alpha - 557.00 = 0 \;\Rightarrow\; \alpha_{0} = 0.13\\
      s_{0} &= \alpha_{0} d_{0} = (1.85,\ 2.52)^T, \quad x_{1} = x_{0} + s_{0} = (1.85,\ 2.52)^T\\
      y_{0} &= g_{1} - g_{0} = (9.94,\ 21.99)^T\\
      A_{1} &= A_{0} - \frac{A_{0} y_{0} y_{0}^T A_{0}}{y_{0}^T A_{0} y_{0}} + \frac{s_{0} s_{0}^T}{s_{0}^T y_{0}} = \begin{pmatrix} 0.88 & -0.31 \\ -0.31 & 0.26 \end{pmatrix}
    \end{align*}
    """
end

@testset "rank-one iteration block (1(i), k = 1): B form, z line" begin
    tr = rank_one(P1i, [0.0, 0.0]; steps = 2)
    b = latex(tr.steps[2])
    # z₁ = y₁ − B₁s₁ ≈ (4.06, −2.99) − (1.04, −0.77) = (3.02, −2.23);  B₂ = H
    @test occursin(raw"d_{1} &= -B_{1}^{-1} g_{1} = (4.47,\ -2.02)^T\\", b)
    @test occursin(raw"z_{1} &= y_{1} - B_{1} s_{1} = (3.02,\ -2.23)^T\\", b)
    @test occursin(raw"B_{2} &= B_{1} + \frac{z_{1} z_{1}^T}{z_{1}^T s_{1}} = \begin{pmatrix} 4.00 & 1.00 \\ 1.00 & 8.00 \end{pmatrix}", b)
    @test !occursin("A_{", b)
end

@testset "quasi-Newton summary table (DFP, 1(i))" begin
    tr = dfp(P1i, [0.0, 0.0]; steps = 2)
    @test latex_table(tr) == raw"""
    \begin{tabular}{c|ccc}
      $k$ & $x_k$ & $F(x_k)$ & $\alpha_k$ \\ \hline
      0 & $(0.00,\ 0.00)^T$ & $0.00$ & $0.13$ \\
      1 & $(1.85,\ 2.52)^T$ & $-36.90$ & $0.25$ \\
      2 & $(3.00,\ 2.00)^T$ & $-40.00$ & --- \\
    \end{tabular}
    """
end

@testset "whole quasi-Newton trace: blocks, stop sentence, table" begin
    tr = dfp(P1i, [0.0, 0.0]; steps = 2)
    @test latex(tr) == join(latex.(tr.steps), "\n") * "\n" *
                       raw"$g_{2} = 0$, so the method stops at $x_{2} = x^*$." * "\n\n" *
                       raw"\medskip" * "\n" * latex_table(tr)
end
