# Problem (1)(i): F(x, y) = 2x² + xy + 4y² − 14x − 19y,  H = [4 1; 1 8],  x* = (3, 2)
F1i(x)  = 2x[1]^2 + x[1]*x[2] + 4x[2]^2 - 14x[1] - 19x[2]
∇F1i(x) = [4x[1] + x[2] - 14, x[1] + 8x[2] - 19]
const H1i = [4.0 1.0; 1.0 8.0]

using Symbolics: @variables
@variables x y
const P1i = Problem(2x^2 + x*y + 4y^2 - 14x - 19y, [x, y])

"Closed-form exact step for a quadratic — independent oracle for the calculus line search."
α_quadratic(H, g, p) = -dot(g, p) / dot(p, H * p)

@variables x1 x2
const P1ii = Problem(2x1^2 * x2^2 + (x1 - 2)^2 + (x2 - 1)^2, [x1, x2])

# Problem (4): F = 20x₁² + 6x₁x₂ + 5x₂² − 62x₁ − 34x₂,  H = [40 6; 6 10],  x* = H \ (62, 34)
const H4 = [40.0 6.0; 6.0 10.0]
const x4 = H4 \ [62.0, 34.0]
const P4 = Problem(20x1^2 + 6x1*x2 + 5x2^2 - 62x1 - 34x2, [x1, x2])
