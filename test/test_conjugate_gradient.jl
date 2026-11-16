
@testset "Conjugate gradient — Problem 1(i)" begin
    @testset "first step matches hand calculation" begin
        tr = conjugate_gradient(F1i, ∇F1i, [0.0, 0.0];
                                steps = 1, linesearch = calculus_linesearch(P1i))
        s = only(tr.steps)
        @test s.k == 0
        @test s.x == [0.0, 0.0]
        @test s.g ≈ [-14, -19]
        @test s.p ≈ [14, 19]          # p₀ = −g₀
        @test s.α ≈ 557 / 4204        # gᵀg / pᵀHp
        @test s.ls.α == s.α           # line-search working is recorded
        @test length(s.ls.roots) == 1
        @test tr.x ≈ (557 / 4204) .* [14, 19]
    end
end

@testset "Conjugate gradient — Problem 1(i), two steps" begin
    tr = conjugate_gradient(F1i, ∇F1i, [0.0, 0.0];
                            steps = 2, linesearch = calculus_linesearch(P1i))
    s0, s1 = tr.steps
    @test dot(s0.g, s1.g) ≈ 0 atol = 1e-12            # successive gradients orthogonal
    @test dot(s0.p, H1i * s1.p) ≈ 0 atol = 1e-12      # directions H-conjugate
    @test s1.β ≈ dot(s1.g, s1.g) / dot(s0.g, s0.g)   # Fletcher–Reeves
    @test tr.x ≈ [3.0, 2.0]                           # quadratic in ℝ² ⇒ exact after n = 2 steps
end

@testset "Conjugate gradient — Problem 1(i), asked for 3 steps" begin
    tr = conjugate_gradient(F1i, ∇F1i, [0.0, 0.0];
                            steps = 3, linesearch = calculus_linesearch(P1i))
    @test length(tr.steps) == 2          # g₂ = 0 ⇒ no third direction exists
    @test tr.status == :converged
    @test tr.x ≈ [3.0, 2.0]

    tr1 = conjugate_gradient(F1i, ∇F1i, [0.0, 0.0];
                             steps = 1, linesearch = calculus_linesearch(P1i))
    @test tr1.status == :max_steps
end

@testset "Conjugate gradient — Problem 1(i), objective values recorded" begin
    tr = conjugate_gradient(F1i, ∇F1i, [0.0, 0.0];
                            steps = 3, linesearch = calculus_linesearch(P1i))
    @test tr.steps[1].Fx == 0.0                      # F(0, 0)
    @test tr.steps[2].Fx < tr.steps[1].Fx            # descent
    @test tr.Fx ≈ -40.0                              # F(3, 2) = 18 + 6 + 16 − 42 − 38
end

@testset "Conjugate gradient — Problem method" begin
    tr  = conjugate_gradient(P1i, [0, 0]; steps = 3)
    ref = conjugate_gradient(F1i, ∇F1i, [0.0, 0.0]; steps = 3, linesearch = calculus_linesearch(P1i))
    @test tr.x ≈ ref.x ≈ [3.0, 2.0]
    @test [s.α for s in tr.steps] ≈ [s.α for s in ref.steps]
    @test tr.status == :converged
end

@testset "Trace keeps the gradient at the last point, for every method" begin
    for tr in (conjugate_gradient(P1i, [0.0, 0.0]; steps = 1),      # :max_steps
               conjugate_gradient(P1i, [0.0, 0.0]; steps = 3),      # :converged
               steepest_descent(P4, [0.0, 0.0]; steps = 50, xtol = 0.1),   # :small_step
               dfp(P1i, [0.0, 0.0]; steps = 1), broyden(P1i, [0.0, 0.0]; steps = 3),
               newton(P1i, [0.0, 0.0]; steps = 1), newton(P1i, [0.0, 0.0]; steps = 3))
        @test tr.g == P1i.∇F(tr.x) || tr.g == P4.∇F(tr.x)
    end
end
