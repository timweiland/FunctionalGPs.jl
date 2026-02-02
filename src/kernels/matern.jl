using Distances
using KernelFunctions
using Polynomials

export HalfIntegerMaternKernel

"""
    HalfIntegerMaternKernel{P, ND, TL, TD, TP} <: Kernel

A Matérn kernel with half-integer smoothness parameter `ν = P + 1/2`.

The Matérn kernel is a popular choice for GP regression due to its tunable
smoothness. Half-integer values of `ν` admit closed-form expressions as
products of exponentials and polynomials, enabling efficient computation
and analytic derivatives.

# Type Parameters
- `P`: Integer such that `ν = P + 1/2` (e.g., `P=0` gives ν=1/2, `P=1` gives ν=3/2)
- `ND`: Number of input dimensions
- `TL`, `TD`, `TP`: Internal types for lengthscales, distance metric, and polynomial

# Common Variants
| P | ν   | Smoothness        | Equivalent         |
|---|-----|-------------------|--------------------|
| 0 | 1/2 | Not differentiable | Exponential kernel |
| 1 | 3/2 | Once differentiable | —                |
| 2 | 5/2 | Twice differentiable | —               |

# Constructor
    HalfIntegerMaternKernel(p::Int, lengthscales)

Create a Matérn kernel with ν = `p` + 1/2 and the given lengthscales.

# Arguments
- `p`: Non-negative integer determining smoothness (ν = p + 1/2)
- `lengthscales`: Scalar or vector of lengthscales (one per dimension)

# Examples
```julia
# Matérn-5/2 kernel (twice differentiable) in 1D with lengthscale 1.0
k = HalfIntegerMaternKernel(2, [1.0])
k([0.0], [0.5])  # Evaluate kernel

# Matérn-3/2 in 2D with different lengthscales per dimension
k2d = HalfIntegerMaternKernel(1, [0.5, 1.0])

# Compute derivatives (1D only)
dk = derivative(k, 1, 0)  # ∂k/∂x
```

See also: [`derivative`](@ref), [`WendlandKernel`](@ref)
"""
struct HalfIntegerMaternKernel{P, ND, TL, TD, TP} <: KernelFunctions.Kernel
    lengthscales::TL
    dist::TD
    poly::TP
end

kernel_structure(::HalfIntegerMaternKernel) = StationaryKernelTrait()

"""
    HalfIntegerMaternDerivativeEvenKernel{TK, TP, TC} <: Kernel

Stationary kernel arising from an even-order derivative of a 1D Matérn kernel.

When the total derivative order (n + m) is even, the resulting kernel
`∂ⁿ⁺ᵐk/∂xⁿ∂yᵐ` remains stationary: it depends only on `|x - y|`. This
enables efficient Toeplitz-based computations for uniformly-spaced data.

Typically constructed via [`derivative`](@ref) rather than directly.

# Fields
- `base`: The original [`HalfIntegerMaternKernel`](@ref)
- `polynomial`: Polynomial factor in the derivative expression
- `coefficient`: Scalar coefficient including sign and lengthscale factors

See also: [`HalfIntegerMaternDerivativeOddKernel`](@ref), [`derivative`](@ref)
"""
struct HalfIntegerMaternDerivativeEvenKernel{TK, TP, TC} <: KernelFunctions.Kernel
    base::TK
    polynomial::TP
    coefficient::TC
end

"""
    HalfIntegerMaternDerivativeOddKernel{TK, TP, TC, TR} <: Kernel

Signed-stationary kernel arising from an odd-order derivative of a 1D Matérn.

When the total derivative order (n + m) is odd, the resulting kernel has
structure `k(x,y) = sign(x-y) · f(|x-y|)`. This "signed-stationary" form
still admits specialized matrix representations but differs from standard
stationary kernels.

Typically constructed via [`derivative`](@ref) rather than directly.

# Fields
- `base`: The original [`HalfIntegerMaternKernel`](@ref)
- `polynomial`: Polynomial factor in the derivative expression
- `coefficient`: Scalar coefficient including sign and lengthscale factors
- `rho`: Scaled inverse lengthscale `√(2ν)/ℓ`

See also: [`HalfIntegerMaternDerivativeEvenKernel`](@ref), [`derivative`](@ref)
"""
struct HalfIntegerMaternDerivativeOddKernel{TK, TP, TC, TR} <: KernelFunctions.Kernel
    base::TK
    polynomial::TP
    coefficient::TC
    rho::TR
end

kernel_structure(::HalfIntegerMaternDerivativeEvenKernel) = StationaryKernelTrait()
kernel_structure(::HalfIntegerMaternDerivativeOddKernel) = SignedStationaryKernelTrait()

function half_integer_matern_coefficients(p::Int)
    coeffs = [Rational(1, 1)]
    for i in (p - 1):-1:0
        push!(coeffs, coeffs[end] * 2 * (i + 1) // ((p + i + 1) * (p - i)))
    end
    return Tuple(coeffs)
end

function HalfIntegerMaternKernel(p::Int, lengthscales)
    ν = p + 1 // 2
    scale_factor = @. sqrt(2 * ν) / lengthscales
    scale_factor = scale_factor .^ 2
    dist = WeightedEuclidean(scale_factor)
    poly = Polynomial(half_integer_matern_coefficients(p))
    return HalfIntegerMaternKernel{p, length(lengthscales), typeof(lengthscales), typeof(dist), typeof(poly)}(lengthscales, dist, poly)
end

# Kernel evaluation methods
function _exp_poly(k::HalfIntegerMaternKernel, d)
    return exp(-d) * k.poly(d)
end

(k::HalfIntegerMaternKernel)(x, y) = _exp_poly(k, k.dist(x, y))

function (k::HalfIntegerMaternDerivativeEvenKernel)(x, y)
    τ = k.base.dist(x, y)
    return k.coefficient * exp(-τ) * k.polynomial(τ)
end

function (k::HalfIntegerMaternDerivativeOddKernel)(x, y)
    τ = k.base.dist(x, y)
    diff = only(x) - only(y)
    if iszero(τ)
        return zero(diff)
    end
    factor = diff * k.rho / τ
    return k.coefficient * factor * exp(-τ) * k.polynomial(τ)
end
