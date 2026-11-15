import Contour

"""    path(tr::Trace) -> [x₀, x₁, …, xₙ] — every iterate, then the end point."""
path(tr::Trace) = [[s.x for s in tr.steps]; [tr.x]]

"""
    contour_lines(F, xlim, ylim, levels; n = 80) -> [(level, [p₁, p₂, …]), …]

Level sets F(x) = c on an n × n grid over `xlim` × `ylim`, one polyline per piece.
"""
function contour_lines(F, xlim, ylim, levels; n = 80)
    xs, ys = range(xlim...; length = n), range(ylim...; length = n)
    Z = [F([x, y]) for x in xs, y in ys]
    return [(Contour.level(c), [[x, y] for (x, y) in zip(Contour.coordinates(l)...)])
            for c in Contour.levels(Contour.contours(xs, ys, Z, levels)) for l in Contour.lines(c)]
end

"""Legend name and pgfplots style of each method, chosen by step type."""
method_name(::Type{CGStep{SteepestDescent}}) = "Steepest descent"
method_name(::Type{CGStep{FletcherReeves}})  = "Conjugate gradient"
method_name(::Type{NewtonStep})              = "Newton"
method_name(::Type{QNStep{Broyden}})         = "Broyden"
plot_style(::Type{CGStep{SteepestDescent}})  = "blue, thick, mark=*, mark size=1.5pt"
plot_style(::Type{CGStep{FletcherReeves}})   = "red, thick, mark=square*, mark size=1.5pt"
plot_style(::Type{NewtonStep})               = "black!60!green, thick, dashed, mark=triangle*, mark size=2pt"
plot_style(::Type{QNStep{Broyden}})          = "orange, very thick, densely dotted, mark=diamond*, mark size=2pt"

"""One trace as a styled path with its legend entry."""
path_plot(tr::Trace{S}) where {S} =
    "\\addplot[$(plot_style(S))] coordinates {$(coords(path(tr)))};\n\\addlegendentry{$(method_name(S))}"

coords(pts) = join(("($(fmt(p[1]; digits = 3)), $(fmt(p[2]; digits = 3)))" for p in pts), " ")

"""
    latex_plot(F, traces; xlim, ylim, levels) -> String

pgfplots figure: grey contours of F at `levels`, then the path of each trace with the
method's style and legend entry.
"""
function latex_plot(F, traces; xlim, ylim, levels)
    contour = ["\\addplot[gray!40, forget plot] coordinates {$(coords(pts))};"
               for (_, pts) in contour_lines(F, xlim, ylim, levels)]
    return """
    \\begin{tikzpicture}
    \\begin{axis}[axis equal image, xmin=$(xlim[1]), xmax=$(xlim[2]), ymin=$(ylim[1]), ymax=$(ylim[2]),
      xlabel={\$x_1\$}, ylabel={\$x_2\$}, legend pos=outer north east, legend cell align=left]
    $(join(contour, "\n"))
    $(join(path_plot.(traces), "\n"))
    \\end{axis}
    \\end{tikzpicture}
    """
end
