using Test
using OptSolutions
using LinearAlgebra: dot

include("fixtures.jl")

@testset "OptSolutions" begin
    include("test_conjugate_gradient.jl")
    include("test_problem.jl")
    include("test_linesearch.jl")
    include("test_newton.jl")
    include("test_dfp.jl")
    include("test_rank_one.jl")
    include("test_latex.jl")
    include("test_solutions.jl")
end
