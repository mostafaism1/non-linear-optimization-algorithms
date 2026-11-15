module OptSolutions

using LinearAlgebra
using Symbolics

export Problem, Trace, CGStep, conjugate_gradient, calculus_linesearch, LineSearchStep, newton, NewtonStep

include("trace.jl")
include("problem.jl")
include("linesearch.jl")
include("algorithms/conjugate_gradient.jl")
include("algorithms/newton.jl")

end # module OptSolutions
