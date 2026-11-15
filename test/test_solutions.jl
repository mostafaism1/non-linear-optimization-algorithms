@testset "fragment p1i" begin
    tex = p1i()
    tr  = conjugate_gradient(P1i, [0.0, 0.0]; steps = 3)
    @test startswith(tex, raw"\subsection*{Problem 1(i)}")
    @test occursin(raw"F(x, y) = 2x^2 + xy + 4y^2 - 14x - 19y", tex)
    @test occursin(latex(tr), tex)                                  # full CG working
    @test occursin("converges in at most \$n = 2\$ steps", tex)       # theory remark
    @test occursin(raw"x^* = (3.00,\ 2.00)^T, \quad F^* = -40.00", tex)   # actual minimum
end

@testset "write_solutions: fragments + main.tex that compiles" begin
    mktempdir() do dir
        write_solutions(dir)
        @test read(joinpath(dir, "p1i.tex"), String) == p1i()
        main = read(joinpath(dir, "main.tex"), String)
        @test occursin(raw"\input{p1i}", main)
        @test occursin(raw"\usepackage{amsmath}", main)
        if Sys.which("latexmk") !== nothing
            ok = success(Cmd(`latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`; dir))
            @test ok
            @test isfile(joinpath(dir, "main.pdf"))
        end
    end
end

@testset "fragment p1ii" begin
    tex = p1ii()
    tr  = conjugate_gradient(P1ii, [0.0, 0.0]; steps = 3)
    nt  = newton(P1ii, tr.x; steps = 20)
    @test startswith(tex, raw"\subsection*{Problem 1(ii)}")
    @test occursin(raw"F(x_1, x_2) = 2x_1^2 x_2^2 + (x_1 - 2)^2 + (x_2 - 1)^2", tex)
    @test occursin(latex(tr), tex)                                    # full CG working (3 steps)
    @test !occursin("so the method stops", tex)                       # :max_steps, no stop claim
    @test occursin(raw"x^* = (1.95,\ 0.12)^T, \quad F^* = 0.89", tex)  # actual minimum, 2 d.p.
    @test occursin("positive definite", tex)                          # second-order check
    # comparison at 4 d.p., where 2 d.p. hides the difference; wording makes no data claim
    @test occursin(raw"\paragraph{Comparison} (to 4 decimal places):", tex)
    @test !occursin("agree", tex)
    @test occursin(raw"F(x_{3}) = 0.8867", tex)
    @test occursin(raw"F^* = 0.8863", tex)
    @test occursin(raw"\|x_{3} - x^*\| = 0.0122", tex)
end

@testset "write_solutions includes p1ii after p1i" begin
    mktempdir() do dir
        write_solutions(dir)
        @test read(joinpath(dir, "p1ii.tex"), String) == p1ii()
        main = read(joinpath(dir, "main.tex"), String)
        @test occursin("\\input{p1i}\n\\clearpage\n\\input{p1ii}", main)   # each part on a new page
    end
end

@testset "fragment p2: certify DFP, two DFP steps, compare with rank-one" begin
    tex = p2()
    A = dfp(P1i, [0.0, 0.0]; steps = 2)
    B = rank_one(P1i, [0.0, 0.0]; steps = 2)
    @test startswith(tex, raw"\section*{Problem 2}")
    @test !occursin("d_", tex)                                        # direction is p_k, as in the PDF
    # (a) certification: typeset proof of the secant condition
    @test occursin(raw"&= A_k y_k - A_k y_k + s_k = s_k", tex)
    # (b) two DFP steps on 1(i), with the condition checked on the computed numbers
    @test occursin(latex(A), tex)
    @test occursin(raw"A_{1} y_{0} = (1.85,\ 2.52)^T = s_{0}", tex)
    @test occursin(raw"A_{2} y_{1} = (1.15,\ -0.52)^T = s_{1}", tex)
    @test occursin(raw"A_{2} = \begin{pmatrix} 0.26 & -0.03 \\ -0.03 & 0.13 \end{pmatrix} = H^{-1}", tex)
    # (c) rank-one comparison
    @test occursin(raw"&= B_k s_k + y_k - B_k s_k = y_k", tex)
    @test occursin(latex(B), tex)
    @test occursin(raw"B_{2} s_{1} = (4.06,\ -2.99)^T = y_{1}", tex)
    @test occursin(raw"B_{2} = \begin{pmatrix} 4.00 & 1.00 \\ 1.00 & 8.00 \end{pmatrix} = H", tex)
    @test occursin(raw"z_k^T s_k", tex) && occursin("64.02", tex) && occursin("4.61", tex)
end

@testset "write_solutions includes p2 after p1ii" begin
    mktempdir() do dir
        write_solutions(dir)
        @test read(joinpath(dir, "p2.tex"), String) == p2()
        main = read(joinpath(dir, "main.tex"), String)
        @test occursin("\\input{p1ii}\n\\clearpage\n\\input{p2}", main)
    end
end

@testset "fragment p3: Broyden — certify, two steps, third step, comparison" begin
    tex = p3()
    B = broyden(P1i, [0.0, 0.0]; steps = 2)
    @test startswith(tex, raw"\section*{Problem 3}")
    @test !occursin("d_", tex)                                        # direction is p_k, as in the PDF
    # (a) certification
    @test occursin(raw"&= B_k s_k - B_k s_k + y_k = y_k", tex)
    # (b) two steps, checked on the computed numbers
    @test occursin(latex(B), tex)
    @test occursin(raw"B_{1} s_{0} = (9.94,\ 21.99)^T = y_{0}", tex)
    @test occursin(raw"B_{2} = \begin{pmatrix} 4.00 & 1.00 \\ 1.00 & 8.00 \end{pmatrix} = H", tex)
    @test occursin(raw"x_{2} = (2.77,\ 1.84)^T \neq x^*", tex)
    @test occursin(raw"\paragraph{Third step.} In iteration $k=1$, $p_{1} = -g_{1}$", tex)   # plain remark
    @test !occursin("Sherman", tex) && !occursin("orthogonal", tex)     # no derivation
    # third step: Newton step with α = 1 to x*
    @test occursin(raw"\alpha_{2} = 1.00", tex)
    @test occursin(raw"x_{3} = (3.00,\ 2.00)^T = x^*", tex)
    # comparison with Problem 2
    @test occursin(raw"\paragraph{Comparison with Problem 2.}", tex)
end

@testset "write_solutions includes p3 after p2" begin
    mktempdir() do dir
        write_solutions(dir)
        @test read(joinpath(dir, "p3.tex"), String) == p3()
        main = read(joinpath(dir, "main.tex"), String)
        @test occursin("\\input{p2}\n\\clearpage\n\\input{p3}", main)
    end
end

@testset "fragment p4: number of line searches with ‖xₖ₊₁ − xₖ‖ ≤ 0.1" begin
    tex = p4()
    @test startswith(tex, "\\section*{Problem 4}")
    @test occursin("x^* = H^{-1} (62.00,\\ 34.00)^T = (1.14,\\ 2.71)^T", tex)
    @test occursin(latex_table(steepest_descent(P4, [0.0, 0.0]; steps = 50, xtol = 0.1)), tex)
    @test occursin(latex_table(conjugate_gradient(P4, [0.0, 0.0]; steps = 50, xtol = 0.1)), tex)
    @test occursin(latex_table(broyden(P4, [0.0, 0.0]; steps = 50, xtol = 0.1)), tex)
    # Newton: one line search, α₀ = 1
    @test occursin("p_{0} &= -H^{-1} g_{0} = (1.14,\\ 2.71)^T", tex)
    @test occursin("\\alpha_{0} = 1.00", tex)
    # answer table
    for row in ["(i) Steepest descent & 5", "(ii) Conjugate gradient & 2",
                "(iii) Newton & 1", "(iv) Broyden & 3"]
        @test occursin(row, tex)
    end
end

@testset "write_solutions includes p4 after p3" begin
    mktempdir() do dir
        write_solutions(dir)
        @test read(joinpath(dir, "p4.tex"), String) == p4()
        @test occursin("\\input{p3}\n\\clearpage\n\\input{p4}", read(joinpath(dir, "main.tex"), String))
    end
end

@testset "fragment p5: k so that u and v are A-conjugate" begin
    tex = p5()
    @test startswith(tex, "\\section*{Problem 5}")
    @test occursin("A v = (7,\\ 1,\\ 12)^T", tex)                 # exact integers, no decimals
    @test occursin("u^T A v = 14 + 3 + 12k = 17 + 12k = 0", tex)
    @test occursin("k = -\\frac{17}{12} \\approx -1.42", tex)
    @test occursin("u = (2,\\ 3,\\ -\\tfrac{17}{12})^T", tex) # check line: uᵀAv = 0 with this k
    @test occursin("u^T A v = 14 + 3 - 17 = 0", tex)
end

@testset "write_solutions includes p5 after p4" begin
    mktempdir() do dir
        write_solutions(dir)
        @test read(joinpath(dir, "p5.tex"), String) == p5()
        @test occursin("\\input{p4}\n\\clearpage\n\\input{p5}", read(joinpath(dir, "main.tex"), String))
    end
end
