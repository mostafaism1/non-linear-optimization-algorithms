"""Fragments in assignment order: file name (without .tex) => generator."""
const FRAGMENTS = ["p1i" => p1i, "p1ii" => p1ii, "p2" => p2]

const MAIN_TEX = raw"""
\documentclass[11pt]{article}
\usepackage[margin=2.5cm]{geometry}
\usepackage{amsmath}
\title{Optimization Techniques --- Assignment 2\\ Reference Solutions}
\date{}
\begin{document}
\maketitle
\section*{Problem 1}
""" * join(("\\input{$name}" for (name, _) in FRAGMENTS), "\n\\clearpage\n") * raw"""

\end{document}
"""

"""
    write_solutions(dir) -> dir

Write one `.tex` fragment per problem part and `main.tex`, which pulls them in with `\\input`.
"""
function write_solutions(dir)
    mkpath(dir)
    for (name, gen) in FRAGMENTS
        write(joinpath(dir, "$name.tex"), gen())
    end
    write(joinpath(dir, "main.tex"), MAIN_TEX)
    return dir
end
