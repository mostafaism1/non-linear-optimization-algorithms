module OptSolutions

using LinearAlgebra

export Trace, CGStep, conjugate_gradient, quadratic_linesearch

include("trace.jl")
include("linesearch.jl")
include("algorithms/conjugate_gradient.jl")

end # module OptSolutions
