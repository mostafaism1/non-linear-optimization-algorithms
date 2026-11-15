using Printf: Format, format

"""
    fmt(x::Real; digits = 2) -> String

Fixed-point with `digits` decimals. A value that rounds to zero prints unsigned (never "-0.00").
"""
function fmt(x::Real; digits = 2)
    s = format(Format("%.$(digits)f"), x)
    return startswith(s, '-') && all(c -> c in "-0.", s) ? s[2:end] : s
end

"""    fmt(v::AbstractVector; digits = 2) -> String — column vector as `(v₁,\\ v₂)^T`."""
fmt(v::AbstractVector; digits = 2) = "(" * join((fmt(x; digits) for x in v), ",\\ ") * ")^T"
