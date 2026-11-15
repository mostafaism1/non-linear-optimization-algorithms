using Test
using OptSolutions
using LinearAlgebra: dot

include("fixtures.jl")

@testset "OptSolutions" begin
    include("test_conjugate_gradient.jl")
    include("test_problem.jl")
    include("test_linesearch.jl")
end
