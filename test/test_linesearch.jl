using Symbolics: @variables

@testset "calculus_linesearch" begin
    @testset "quadratic 1(i): matches closed form α = −gᵀp / pᵀHp" begin
        @variables x y
        P  = Problem(2x^2 + x*y + 4y^2 - 14x - 19y, [x, y])
        ls = calculus_linesearch(P)
        for (x0, p) in (([0.0, 0.0], [14.0, 19.0]), ([1.0, -2.0], [0.3, 1.7]))
            g = P.∇F(x0)
            r = ls(x0, p)
            @test r.α ≈ α_quadratic(H1i, g, p)
            @test length(r.roots) == 1               # φ′ linear ⇒ one root
        end
    end

    @testset "quartic: picks global minimizer of φ over α > 0" begin
        @variables t
        P2 = Problem(t^4 - 2t^2 + 0.5t, [t])
        x0, p = [0.0], [-1.0]                        # φ(α) = α⁴ − 2α² − 0.5α
        r  = calculus_linesearch(P2)(x0, p)
        φ(α) = P2.F(x0 + α * p)
        @test length(r.roots) == 3                   # φ′ cubic with three real roots
        @test r.α > 0
        @test all(φ(r.α) ≤ φ(s) + 1e-12 for s in 0:1e-3:3)
    end
end
