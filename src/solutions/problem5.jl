"""Problem 5: find k so that u = (2, 3, k)ᵀ and v = (1, -1, 3)ᵀ are A-conjugate (uᵀAv = 0)."""
function p5(; digits = 2)
    A  = [5 1 1; 1 3 1; 1 1 4]
    v  = [1, -1, 3]
    Av = A * v
    c  = dot([2, 3], Av[1:2])                 # uᵀAv = c + k (Av)₃, linear in k
    k  = -c // Av[3]
    ints(w) = "(" * join(w, ",\\ ") * ")^T"
    frac(q; f = "\\frac") = (q < 0 ? "-" : "") * "$f{$(abs(numerator(q)))}{$(denominator(q))}"
    signed(t) = t < 0 ? "- $(-t)" : "+ $t"
    check = "$(2Av[1]) $(signed(3Av[2])) $(signed(Int(k * Av[3])))"
    return """
    \\section*{Problem 5}

    Find \$k\$ so that \$u = (2,\\ 3,\\ k)^T\$ and \$v = (1,\\ -1,\\ 3)^T\$ are conjugate with respect to
    \\[ A = $(fmt(A; digits = 0)). \\]
    Two vectors are \$A\$-conjugate when \$u^T A v = 0\$. First,
    \\[ A v = $(ints(Av)), \\]
    so
    \\[ u^T A v = $(2Av[1]) + $(3Av[2]) + $(Av[3])k = $c + $(Av[3])k = 0
       \\quad\\Rightarrow\\quad k = $(frac(k)) \\approx $(fmt(float(k); digits)). \\]

    \\paragraph{Check.} With \$u = (2,\\ 3,\\ $(frac(k; f = "\\tfrac")))^T\$:
    \\[ u^T A v = $check = 0. \\]
    """
end
