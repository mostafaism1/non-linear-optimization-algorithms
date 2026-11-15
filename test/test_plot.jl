@testset "path: x₀, x₁, …, xₙ of a trace" begin
    sd = steepest_descent(P4, [0.0, 0.0]; steps = 50, xtol = 0.1)
    pts = path(sd)
    @test length(pts) == length(sd.steps) + 1
    @test pts[1] == [0.0, 0.0] && pts[2] == sd.steps[2].x && pts[end] == sd.x
    @test path(newton(P4, [0.0, 0.0]; steps = 50)) ≈ [[0.0, 0.0], x4]   # one step to x*
end

@testset "contour_lines: every point lies on its level" begin
    levels = [-80.0, -60.0, -20.0]
    lines  = contour_lines(P4.F, (-0.5, 2.5), (-0.5, 3.5), levels)
    @test sort(unique(first.(lines))) == levels                 # each level gives a line
    @test all(abs(P4.F(p) - c) < 0.1 for (c, pts) in lines for p in pts)
end

@testset "latex_plot: contours + one styled, labelled path per trace" begin
    sd  = steepest_descent(P4, [0.0, 0.0]; steps = 50, xtol = 0.1)
    cg  = conjugate_gradient(P4, [0.0, 0.0]; steps = 50, xtol = 0.1)
    nt  = newton(P4, [0.0, 0.0]; steps = 50)
    br  = broyden(P4, [0.0, 0.0]; steps = 50, xtol = 0.1)
    tex = latex_plot(P4.F, [sd, cg, nt, br]; xlim = (-0.5, 2.5), ylim = (-0.5, 3.5), levels = [-80.0, -40.0])
    @test startswith(tex, "\\begin{tikzpicture}")
    @test occursin("\\begin{axis}[", tex) && occursin("axis equal image", tex)
    @test occursin("width=\\linewidth", tex) && occursin("legend pos=south east", tex)   # full width, legend inside
    @test count("forget plot", tex) == length(contour_lines(P4.F, (-0.5, 2.5), (-0.5, 3.5), [-80.0, -40.0]))
    sd_coords = join(("($(fmt(p[1]; digits = 3)), $(fmt(p[2]; digits = 3)))" for p in path(sd)), " ")
    @test occursin("coordinates {$sd_coords};", tex)
    for name in ["Steepest descent", "Conjugate gradient", "Newton", "Broyden"]
        @test count("\\addlegendentry{$name}", tex) == 1
    end
    @test endswith(tex, "\\end{tikzpicture}\n")
end

@testset "latex_plot: CG and the quasi-Newton methods on 1(i), distinct styles" begin
    trs = [conjugate_gradient(P1i, [0.0, 0.0]; steps = 3), dfp(P1i, [0.0, 0.0]; steps = 2),
           rank_one(P1i, [0.0, 0.0]; steps = 2), broyden(P1i, [0.0, 0.0]; steps = 3)]
    tex = latex_plot(P1i.F, trs; xlim = (-0.5, 4.0), ylim = (-0.5, 3.0), levels = [-30.0])
    for name in ["Conjugate gradient", "DFP", "Rank-one", "Broyden"]
        @test count("\\addlegendentry{$name}", tex) == 1
    end
    types = [CGStep{SteepestDescent}, CGStep{FletcherReeves}, NewtonStep,
             QNStep{DFP}, QNStep{RankOne}, QNStep{Broyden}]
    @test allunique(OptSolutions.plot_style.(types))
end

@testset "1(i): DFP and rank-one give the same points as CG (claim in the p2 text)" begin
    cg = path(conjugate_gradient(P1i, [0.0, 0.0]; steps = 3))
    @test path(dfp(P1i, [0.0, 0.0]; steps = 2)) ≈ cg
    @test path(rank_one(P1i, [0.0, 0.0]; steps = 2)) ≈ cg
end
