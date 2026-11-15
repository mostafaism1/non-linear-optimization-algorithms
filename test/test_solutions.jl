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
