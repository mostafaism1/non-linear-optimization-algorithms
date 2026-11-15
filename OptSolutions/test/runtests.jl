using Test
using OptSolutions
using LinearAlgebra: dot

@testset "OptSolutions" begin
    include("test_conjugate_gradient.jl")
end
