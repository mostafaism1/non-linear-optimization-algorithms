module OptSolutions

using LinearAlgebra
using Symbolics

export Problem, Trace, CGStep, conjugate_gradient, calculus_linesearch, LineSearchStep, newton, NewtonStep, fmt, fmt_poly, latex, latex_table, p1i

include("trace.jl")
include("problem.jl")
include("linesearch.jl")
include("algorithms/conjugate_gradient.jl")
include("algorithms/newton.jl")
include("latex.jl")
include("solutions/problem1.jl")

end # module OptSolutions
