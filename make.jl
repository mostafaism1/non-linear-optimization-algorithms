# Regenerate the LaTeX solutions:  julia --project=. make.jl   (then: cd tex && latexmk -pdf main.tex)
using OptSolutions
write_solutions(joinpath(@__DIR__, "tex"))
