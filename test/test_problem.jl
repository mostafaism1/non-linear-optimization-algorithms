using Symbolics: @variables

@testset "Problem — derived from symbolic F" begin
    @variables x y
    P = Problem(2x^2 + x*y + 4y^2 - 14x - 19y, [x, y])

    for pt in ([0.0, 0.0], [1.0, 2.0], [-0.5, 3.25])
        @test P.F(pt) ≈ F1i(pt)
        @test P.∇F(pt) ≈ ∇F1i(pt)
        @test P.H(pt) ≈ H1i
    end
end
