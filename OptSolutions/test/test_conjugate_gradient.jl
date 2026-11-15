# Problem (1)(i): F(x, y) = 2x² + xy + 4y² − 14x − 19y,  H = [4 1; 1 8],  x* = (3, 2)
const F1i(x)  = 2x[1]^2 + x[1]*x[2] + 4x[2]^2 - 14x[1] - 19x[2]
const ∇F1i(x) = [4x[1] + x[2] - 14, x[1] + 8x[2] - 19]
const H1i     = [4.0 1.0; 1.0 8.0]

@testset "Conjugate gradient — Problem 1(i)" begin
    @testset "first step matches hand calculation" begin
        tr = conjugate_gradient(F1i, ∇F1i, [0.0, 0.0];
                                steps = 1, linesearch = quadratic_linesearch(H1i))
        s = only(tr.steps)
        @test s.k == 0
        @test s.x == [0.0, 0.0]
        @test s.g ≈ [-14, -19]
        @test s.p ≈ [14, 19]          # p₀ = −g₀
        @test s.α ≈ 557 / 4204        # gᵀg / pᵀHp
        @test tr.x ≈ (557 / 4204) .* [14, 19]
    end
end

@testset "Conjugate gradient — Problem 1(i), two steps" begin
    tr = conjugate_gradient(F1i, ∇F1i, [0.0, 0.0];
                            steps = 2, linesearch = quadratic_linesearch(H1i))
    s0, s1 = tr.steps
    @test dot(s0.g, s1.g) ≈ 0 atol = 1e-12            # successive gradients orthogonal
    @test dot(s0.p, H1i * s1.p) ≈ 0 atol = 1e-12      # directions H-conjugate
    @test s1.β ≈ dot(s1.g, s1.g) / dot(s0.g, s0.g)   # Fletcher–Reeves
    @test tr.x ≈ [3.0, 2.0]                           # quadratic in ℝ² ⇒ exact after n = 2 steps
end

@testset "Conjugate gradient — Problem 1(i), asked for 3 steps" begin
    tr = conjugate_gradient(F1i, ∇F1i, [0.0, 0.0];
                            steps = 3, linesearch = quadratic_linesearch(H1i))
    @test length(tr.steps) == 2          # g₂ = 0 ⇒ no third direction exists
    @test tr.status == :converged
    @test tr.x ≈ [3.0, 2.0]

    tr1 = conjugate_gradient(F1i, ∇F1i, [0.0, 0.0];
                             steps = 1, linesearch = quadratic_linesearch(H1i))
    @test tr1.status == :max_steps
end
