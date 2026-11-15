stepnorm(s::CGStep) = norm(s.α * s.p)
stepnorm(s::QNStep) = norm(s.s)

@testset "Steepest descent — pₖ = −gₖ, exact line search" begin
    tr = steepest_descent(P1i, [0.0, 0.0]; steps = 2)
    for s in tr.steps
        @test s.p ≈ -s.g
        @test s.α ≈ α_quadratic(H1i, s.g, s.p)
    end
    @test dot(tr.steps[1].p, tr.steps[2].p) ≈ 0 atol = 1e-9   # zig-zag: successive steps ⟂
end

@testset "Problem 4 — stop when ‖xₖ₊₁ − xₖ‖ ≤ 0.1" begin
    sd = steepest_descent(P4, [0.0, 0.0]; steps = 50, xtol = 0.1)
    @test length(sd.steps) == 5
    @test sd.status == :small_step
    @test stepnorm(sd.steps[end]) ≤ 0.1
    @test all(stepnorm(s) > 0.1 for s in sd.steps[1:end-1])

    cg = conjugate_gradient(P4, [0.0, 0.0]; steps = 50, xtol = 0.1)
    @test length(cg.steps) == 2
    @test cg.status == :converged
    @test cg.x ≈ x4

    br = broyden(P4, [0.0, 0.0]; steps = 50, xtol = 0.1)
    @test length(br.steps) == 3
    @test br.status == :converged
    @test br.x ≈ x4

    nt = newton(P4, [0.0, 0.0]; steps = 50)
    @test length(nt.steps) == 1
    @test calculus_linesearch(P4)(nt.steps[1].x, nt.steps[1].d).α ≈ 1   # line search on −H⁻¹g gives α = 1
    @test nt.x ≈ x4
end

@testset "The β rule is part of the step type" begin
    @test conjugate_gradient(P1i, [0.0, 0.0]; steps = 1) isa Trace{CGStep{FletcherReeves}}
    @test steepest_descent(P1i, [0.0, 0.0]; steps = 1) isa Trace{CGStep{SteepestDescent}}
    @test FletcherReeves()([3.0, 4.0], [0.0, 5.0]) ≈ 1.0
    @test SteepestDescent()([3.0, 4.0], [0.0, 5.0]) == 0.0
end
