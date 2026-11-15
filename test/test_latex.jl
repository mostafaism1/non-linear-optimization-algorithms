@testset "latex formatting" begin
    @testset "numbers: fixed 2 d.p. by default" begin
        @test fmt(0.132493) == "0.13"
        @test fmt(-14) == "-14.00"
        @test fmt(-0.001) == "0.00"                # no "-0.00"
        @test fmt(0.88634; digits = 4) == "0.8863" # extra precision on demand
    end

    @testset "vectors: column vector as transposed tuple" begin
        @test fmt([1.8549, 2.5174]) == raw"(1.85,\ 2.52)^T"
        @test fmt([-14.0, -19.0]; digits = 1) == raw"(-14.0,\ -19.0)^T"
    end
end
