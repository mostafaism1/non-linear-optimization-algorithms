module OptSolutions

using LinearAlgebra
using Symbolics

export Problem, Trace, CGStep, conjugate_gradient, calculus_linesearch, LineSearchStep, newton, NewtonStep, dfp, DFPStep, rank_one, RankOneStep, fmt, fmt_poly, latex, latex_table, p1i, p1ii, write_solutions

include("trace.jl")
include("problem.jl")
include("linesearch.jl")
include("algorithms/conjugate_gradient.jl")
include("algorithms/newton.jl")
include("algorithms/dfp.jl")
include("algorithms/rank_one.jl")
include("latex.jl")
include("solutions/problem1.jl")
include("solutions/write.jl")

end # module OptSolutions
