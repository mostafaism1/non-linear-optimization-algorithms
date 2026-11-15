using LinearAlgebra: I

@testset "Rank-one update — Problem 1(i), two steps from B₀ = I" begin
    tr = rank_one(P1i, [0.0, 0.0]; steps = 2)
    s0, s1 = tr.steps

    @test s0.k == 0 && s1.k == 1
    @test s0.x == [0.0, 0.0]
    @test s0.Fx == 0.0
    @test s0.g ≈ [-14, -19]
    @test s0.M == Matrix(1.0I, 2, 2)                  # B₀ = I
    @test s0.p ≈ [14, 19]                             # p₀ = −B₀⁻¹g₀ = −g₀
    @test s0.α ≈ α_quadratic(H1i, s0.g, s0.p)         # exact line search
    @test s0.ls.α == s0.α
    for s in tr.steps
        @test s.M * s.p ≈ -s.g                        # pₖ = −Bₖ⁻¹gₖ
        @test s.s ≈ s.α * s.p
        @test s.y ≈ ∇F1i(s.x + s.s) - s.g
        z = s.y - s.M * s.s                           # z = yₖ − Bₖsₖ
        @test s.M₊ ≈ s.M + z * z' / dot(z, s.s)        # rank-one update
        @test s.M₊ * s.s ≈ s.y                        # quasi-Newton condition
    end
    @test s1.x ≈ s0.x + s0.s
    @test s1.M ≈ s0.M₊
    @test tr.x ≈ [3.0, 2.0]
    @test s1.M₊ ≈ H1i                                 # B₂ = H on a quadratic
    @test tr.status == :converged
end
