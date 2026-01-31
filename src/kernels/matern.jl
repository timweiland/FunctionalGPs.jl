using Distances
using KernelFunctions
using Polynomials

export HalfIntegerMaternKernel

"""
    ν = (P + 1//2) Matern
"""
struct HalfIntegerMaternKernel{P, ND, TL, TD, TP} <: KernelFunctions.Kernel
    lengthscales::TL
    dist::TD
    poly::TP
end

kernel_structure(::HalfIntegerMaternKernel) = StationaryKernelTrait()

"""
    HalfIntegerMaternDerivativeEvenKernel

Stationary kernel arising from taking an even-order derivative of a 1D
half-integer Matérn kernel. The kernel remains stationary up to a global sign
and therefore benefits from the lazy stationary machinery.
"""
struct HalfIntegerMaternDerivativeEvenKernel{TK, TP, TC} <: KernelFunctions.Kernel
    base::TK
    polynomial::TP
    coefficient::TC
end

"""
    HalfIntegerMaternDerivativeOddKernel

Kernel representing an odd-order mixed derivative of a 1D half-integer Matérn.
The sign depends on the input ordering, so it cannot exploit stationary
structure directly.
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
