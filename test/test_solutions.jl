@testset "fragment p1i" begin
    tex = p1i()
    tr  = conjugate_gradient(P1i, [0.0, 0.0]; steps = 3)
    @test startswith(tex, raw"\section*{Problem 1}")                  # Problem 1 heading opens part (i)
    @test occursin(raw"F(x, y) = 2x^2 + xy + 4y^2 - 14x - 19y", tex)
    @test occursin(latex(tr), tex)                                  # full CG working
    @test occursin("converges in at most \$n = 2\$ steps", tex)       # theory remark
    @test occursin(raw"x^* = (3.00,\ 2.00)^T, \quad F^* = -40.00", tex)   # actual minimum
    @test occursin("\\addlegendentry{Conjugate gradient}", tex)       # path plot
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
    @test startswith(tex, raw"\subsection*{(ii) $F(x_1, x_2) = 2x_1^2 x_2^2 + (x_1 - 2)^2 + (x_2 - 1)^2$}")
    @test occursin(raw"F(x_1, x_2) = 2x_1^2 x_2^2 + (x_1 - 2)^2 + (x_2 - 1)^2", tex)
    @test occursin(latex(tr), tex)                                    # full CG working (3 steps)
    @test !occursin("so the method stops", tex)                       # :max_steps, no stop claim
    @test occursin(raw"x^* = (1.95,\ 0.12)^T, \quad F^* = 0.89", tex)  # actual minimum, 2 d.p.
    @test occursin("positive definite", tex)                          # second-order check
    # comparison at 4 d.p., where 2 d.p. hides the difference; wording makes no data claim
    @test occursin("to 4 decimal places", tex)
    @test !occursin("agree", tex)
    @test occursin(raw"F(x_{3}) = 0.8867", tex)
    @test occursin(raw"F^* = 0.8863", tex)
    @test occursin(raw"\|x_{3} - x^*\| = 0.0122", tex)
    @test occursin("\\addlegendentry{Conjugate gradient}", tex)       # path plot
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
    @test occursin(raw"\paragraph{(iii) Comments.} DFP and the rank-one update reach $x^*$ in $n = 2$ steps.", tex)
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
    @test !occursin("H^{-1} (62.00", tex)                  # x* is not asked for; Newton finds it in (iii)
    @test occursin(raw"F_{x_1} &= 40x_1 + 6x_2 - 62", tex) && occursin(raw"F_{x_2} &= 6x_1 + 10x_2 - 34", tex)
    @test occursin(raw"g = \nabla F &= (40x_1 + 6x_2 - 62,\ 6x_1 + 10x_2 - 34)^T", tex)
    @test occursin(raw"H = \nabla^2 F &= \begin{pmatrix} 40 & 6 \\ 6 & 10 \end{pmatrix}", tex)
    @test occursin(raw"Stop after the search that gives $\|x_{k+1} - x_k\| \le 0.1$, or when $g_{k+1} = 0$.", tex)
    @test !occursin("Each one-dimensional search is exact", tex)
    @test occursin(latex(steepest_descent(P4, [0.0, 0.0]; steps = 50, xtol = 0.1)), tex)   # full working + table
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

@testset "p4 has the path plot; main.tex loads pgfplots" begin
    tex = p4()
    @test occursin("\\begin{tikzpicture}", tex) && occursin("\\addlegendentry{Broyden}", tex)
    mktempdir() do dir
        write_solutions(dir)
        @test occursin("\\usepackage{pgfplots}", read(joinpath(dir, "main.tex"), String))
    end
end

@testset "p2 and p3 have path plots" begin
    p2tex, p3tex = p2(), p3()
    @test occursin("\\addlegendentry{DFP}", p2tex) && occursin("\\addlegendentry{Rank-one}", p2tex)
    @test occursin("\\addlegendentry{Broyden}", p3tex) && occursin("\\addlegendentry{DFP}", p3tex)
end

@testset "title page: course code, name and assignment number only" begin
    mktempdir() do dir
        write_solutions(dir)
        main = read(joinpath(dir, "main.tex"), String)
        @test occursin(raw"\documentclass[11pt,titlepage]{article}", main)     # title on its own page
        @test occursin(raw"\title{MTH-645 Optimization Techniques\\ Assignment (2)}", main)
        @test occursin(raw"\author{}", main) && occursin(raw"\date{}", main)
    end
end

@testset "each problem repeats the assignment text, then the Solution marker" begin
    marker = raw"\begin{center}\textbf{Solution:}\end{center}"
    texts = [p1i() => "Comment on your result after the three steps in comparison with the actual minimum point.",
             p2()  => "Certify that the following Davidon--Fletcher--Powell (DFP) updating formula",
             p3()  => "Repeat (2) above using the Broyden quasi-Newton method",
             p4()  => "What is the expected number of one-dimensional",
             p5()  => "w.r.t.\\ the positive definite matrix"]
    for (tex, phrase) in texts
        @test occursin(phrase, tex)
        @test count(marker, tex) == 1
        @test findfirst(phrase, tex).start < findfirst(marker, tex).start
    end
    @test !occursin(marker, p1ii())                                  # one statement for Problem 1
    mktempdir() do dir
        write_solutions(dir)
        @test !occursin("Problem 1}", read(joinpath(dir, "main.tex"), String))   # heading moved to p1i
    end
end

@testset "part labels (i), (ii), (iii) follow the assignment" begin
    @test occursin(raw"\subsection*{(i) $F(x, y) = 2x^2 + xy + 4y^2 - 14x - 19y$}", p1i())
    p2tex, p3tex = p2(), p3()
    for h in ["(i) Quasi-Newton condition.", "(ii) Two DFP steps on Problem 1(i).",
              "(iii) Comparison with the rank-one update."]
        @test occursin("\\paragraph{$h}", p2tex)
    end
    for h in ["(i) Quasi-Newton condition.", "(ii) Two Broyden steps on Problem 1(i)."]
        @test occursin("\\paragraph{$h}", p3tex)
    end
    @test !any(occursin("\\paragraph{($c)", p2tex * p3tex) for c in "abc")
end

@testset "the gradient is written out before the iterations" begin
    g1i = raw"(4x + y - 14,\ x + 8y - 19)^T"
    @test occursin(raw"F_x &= 4x + y - 14", p1i()) && occursin(raw"F_y &= x + 8y - 19", p1i())
    @test occursin(raw"g = \nabla F &= " * g1i, p1i())
    tex = p1ii()
    @test occursin(raw"F_{x_1} &= 4x_1 x_2^2 + 2(x_1 - 2)", tex)
    @test occursin(raw"F_{x_2} &= 4x_1^2 x_2 + 2(x_2 - 1)", tex)
    @test occursin(raw"g = \nabla F &= (4x_1 x_2^2 + 2(x_1 - 2),\ 4x_1^2 x_2 + 2(x_2 - 1))^T", tex)
    @test occursin(raw"g = \nabla F = " * g1i, p2()) && occursin(raw"g = \nabla F = " * g1i, p3())
end

@testset "the Hessian is derived wherever it is used" begin
    H1i = raw"H = \nabla^2 F"
    @test occursin(H1i * raw" &= \begin{pmatrix} 4 & 1 \\ 1 & 8 \end{pmatrix}", p1i())
    for tex in (p2(), p3())                       # "As in Problem 1(i)", before the first step
        @test occursin(H1i * raw" = \begin{pmatrix} 4 & 1 \\ 1 & 8 \end{pmatrix}", tex)
        @test findfirst(H1i, tex).start < findfirst(raw"Iteration $k=0$", tex).start
    end
    tex = p1ii()
    @test occursin(H1i * raw" &= \begin{pmatrix} 4x_2^2 + 2 & 8x_1 x_2 \\ 8x_1 x_2 & 4x_1^2 + 2 \end{pmatrix}", tex)
    @test occursin(raw"H(x^*) = \begin{pmatrix}", tex)
end

@testset "one closing Comments heading in Problems 1 to 3" begin
    for tex in (p1i(), p1ii(), p2(), p3())
        @test count("Comments.}", tex) == 1
        @test !occursin(raw"\paragraph{Remark.}", tex) && !occursin(raw"\paragraph{Comparison", tex)
    end
    tex = p1i()
    @test findfirst("Actual minimum.", tex).start < findfirst("Comments.", tex).start
end

@testset "certify: each secant-condition derivation is a proof (amsthm)" begin
    inside_proof(tex, line) = any(m -> occursin(line, m.match), eachmatch(r"\\begin\{proof\}.*?\\end\{proof\}"s, tex))
    p2tex, p3tex = p2(), p3()
    @test count("\\begin{proof}", p2tex) == 2 && count("\\begin{proof}", p3tex) == 1
    @test inside_proof(p2tex, raw"&= A_k y_k - A_k y_k + s_k = s_k")         # DFP
    @test inside_proof(p2tex, raw"&= B_k s_k + y_k - B_k s_k = y_k")         # rank-one
    @test inside_proof(p3tex, raw"&= B_k s_k - B_k s_k + y_k = y_k")         # Broyden
    mktempdir() do dir
        write_solutions(dir)
        @test occursin(raw"\usepackage{amsthm}", read(joinpath(dir, "main.tex"), String))
    end
end

@testset "p4: CG and Newton start the next iteration to show that it does not move" begin
    tex = p4()
    @test occursin(raw"g_{2} &= \nabla F(x_{2}) = (0.00,\ 0.00)^T", tex)         # CG
    @test occursin(raw"p_{2} &= -g_{2} + \beta_{2} p_{1} = (0.00,\ 0.00)^T", tex)
    @test occursin(raw"$p_{2} = 0$, so the step does not move: $x_{3} = x_{2} = x^*$.", tex)
    @test occursin(raw"g_{1} &= \nabla F(x_{1}) = (0.00,\ 0.00)^T", tex)         # Newton
    @test occursin(raw"p_{1} &= -H^{-1} g_{1} = (0.00,\ 0.00)^T", tex)
    @test occursin(raw"$p_{1} = 0$, so the step does not move: $x_{2} = x_{1} = x^*$.", tex)
    @test !occursin(raw"$g_{2} = 0$", tex) && !occursin(raw"$g_{1} = 0$", tex)
end

@testset "CG is spelled in full at its first use in each problem" begin
    for tex in (p1i(), p4())
        @test findfirst("CG", tex).start == findfirst("(CG)", tex).start + 1   # first CG is the definition
    end
    @test occursin("as the conjugate gradient method in Problem 1(i)", p2())  # one use only: no abbreviation
    @test !occursin("CG", p2()) && !occursin("CG", p3())
end

@testset "each figure has a caption below it, on the same page" begin
    for tex in (p1i(), p1ii(), p2(), p3(), p4())
        m = match(r"\\begin\{minipage\}\{\\linewidth\}(.*?)\\end\{minipage\}"s, tex)
        @test m !== nothing
        @test occursin(r"\\end\{tikzpicture\}.*\\captionof\{figure\}\{Paths? of "s, m[1])
        @test !occursin(raw"\paragraph{Path", tex)
    end
    @test occursin(raw"\usepackage{caption}", OptSolutions.MAIN_TEX)
    @test occursin(raw"\captionsetup{labelfont=bf}", OptSolutions.MAIN_TEX)      # bold "Figure n:"
    @test startswith(OptSolutions.captioned("P\n", "C"), raw"\bigskip")             # space above the plot
end

@testset "p3 comments: plain sentences, no list" begin
    tex = p3()
    @test !occursin(raw"\begin{itemize}", tex)
    @test occursin("Broyden\nneeds 3, because", tex)
end
