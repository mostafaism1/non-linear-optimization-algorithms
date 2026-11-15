using LinearAlgebra: isposdef, norm

@testset "newton" begin
    @testset "quadratic 1(i): one step lands on x* (theory)" begin
        tr = newton(P1i, [0.0, 0.0]; steps = 5)
        @test length(tr.steps) == 1
        s = tr.steps[1]
        @test s.k == 0
        @test s.x == [0.0, 0.0]
        @test s.Fx == 0.0
        @test s.g ≈ [-14.0, -19.0]
        @test s.H ≈ H1i
        @test s.d ≈ [3.0, 2.0]                     # d = −H⁻¹g
        @test tr.x ≈ [3.0, 2.0]
        @test tr.Fx ≈ -40.0
        @test tr.status == :converged
    end

    @testset "status :max_steps when budget runs out" begin
        tr = newton(P1ii, [0.0, 0.0]; steps = 1)
        @test tr.status == :max_steps
    end

    @testset "1(ii): from the CG endpoint, reaches a strict local minimum" begin
        cg = conjugate_gradient(P1ii, [0.0, 0.0]; steps = 3)
        tr = newton(P1ii, cg.x; steps = 20)
        @test tr.status == :converged
        @test norm(P1ii.∇F(tr.x)) ≤ 1e-10
        @test isposdef(P1ii.H(tr.x))
        @test tr.Fx ≤ cg.Fx
    end
end
