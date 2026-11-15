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
