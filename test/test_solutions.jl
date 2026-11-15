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
