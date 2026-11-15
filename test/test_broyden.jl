using LinearAlgebra: I

@testset "Broyden update — Problem 1(i), two steps from B₀ = I" begin
    tr = broyden(P1i, [0.0, 0.0]; steps = 2)
    s0, s1 = tr.steps

    @test s0.x == [0.0, 0.0]
    @test s0.M == Matrix(1.0I, 2, 2)                  # B₀ = I
    @test s0.p ≈ [14, 19]                             # p₀ = −g₀
    @test s0.α ≈ α_quadratic(H1i, s0.g, s0.p)         # exact line search
    for s in tr.steps
        @test s.M * s.p ≈ -s.g                        # pₖ = −Bₖ⁻¹gₖ
        @test s.s ≈ s.α * s.p
        @test s.y ≈ ∇F1i(s.x + s.s) - s.g
        @test s.M₊ ≈ s.M - (s.M * s.s - s.y) * s.s' / dot(s.s, s.s)   # Broyden update
        @test s.M₊ * s.s ≈ s.y                        # quasi-Newton condition
    end
    @test s1.M ≈ s0.M₊
    # Sherman–Morrison with s₀ᵀg₁ = 0: p₁ = −B₁⁻¹g₁ = −g₁, so s₀ ⟂ s₁ and B₂ = H in R²
    @test s1.p ≈ -s1.g
    @test abs(dot(s0.s, s1.s)) < 1e-9
    @test s1.M₊ ≈ H1i
    # but two steps do not reach x* (unlike DFP and rank-one)
    @test !(tr.x ≈ [3.0, 2.0])
    @test tr.status == :max_steps
end

@testset "Broyden — third step is a Newton step to x*" begin
    tr = broyden(P1i, [0.0, 0.0]; steps = 3)
    @test tr.steps[3].α ≈ 1.0
    @test tr.x ≈ [3.0, 2.0]
    @test tr.status == :converged
end
