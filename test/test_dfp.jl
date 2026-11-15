using LinearAlgebra: I

@testset "DFP — Problem 1(i), two steps from A₀ = I" begin
    tr = dfp(P1i, [0.0, 0.0]; steps = 2)
    s0, s1 = tr.steps

    @test s0.k == 0 && s1.k == 1
    @test s0.x == [0.0, 0.0]
    @test s0.Fx == 0.0
    @test s0.g ≈ [-14, -19]
    @test s0.M == Matrix(1.0I, 2, 2)                  # A₀ = I
    @test s0.p ≈ [14, 19]                             # p₀ = −A₀g₀ = −g₀
    @test s0.α ≈ α_quadratic(H1i, s0.g, s0.p)         # exact line search
    @test s0.ls.α == s0.α
    for s in tr.steps
        @test s.p ≈ -s.M * s.g
        @test s.s ≈ s.α * s.p                         # sₖ = xₖ₊₁ − xₖ
        @test s.y ≈ ∇F1i(s.x + s.s) - s.g             # yₖ = gₖ₊₁ − gₖ
        @test s.M₊ * s.y ≈ s.s                        # quasi-Newton condition
    end
    @test s1.x ≈ s0.x + s0.s
    @test s1.M ≈ s0.M₊
    @test tr.x ≈ [3.0, 2.0]                           # quadratic, exact searches ⇒ n = 2 steps
    @test s1.M₊ ≈ inv(H1i)                            # A₂ = H⁻¹ on a quadratic
    @test tr.status == :converged
end
