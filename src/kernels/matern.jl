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

function _half_integer_matern_rho(k::HalfIntegerMaternKernel{P}) where {P}
    ℓ = only(k.lengthscales)
    return sqrt(float(2 * P + 1)) / float(ℓ)
end

function _float_polynomial(poly::Polynomial)
    return Polynomial(float.(coeffs(poly)))
end

function _matern_derivative_polynomial(poly::Polynomial, order::Int)
    result = poly
    for _ in 1:order
        result = Polynomials.derivative(result) - result
    end
    return result
end

function _derivative_polynomial(k::HalfIntegerMaternKernel, total_order::Int)
    float_poly = _float_polynomial(k.poly)
    return _matern_derivative_polynomial(float_poly, total_order)
end

function stationary_kernel_spec(
        k::HalfIntegerMaternKernel{P},
        ::Type{T},
    ) where {P, T <: AbstractFloat}
    sqrt_2nu = sqrt(T(2 * P + 1))
    scales = collect(sqrt_2nu ./ T.(k.lengthscales))
    radial_map = let kernel = k, zero_T = zero(T)
        r2 -> begin
            r = sqrt(max(r2, zero_T))
            return convert(T, _exp_poly(kernel, r))
        end
    end
    return StationaryKernelSpec(scales, radial_map)
end

function stationary_kernel_spec(
        k::HalfIntegerMaternDerivativeOddKernel,
        ::Type{T},
    ) where {T <: AbstractFloat}
    base_spec = stationary_kernel_spec(k.base, T)
    base_spec === nothing && return nothing
    coeff_T = T(k.coefficient)
    signed_map = let kernel = k, coeff = coeff_T, zero_T = zero(T)
        (r2, s) -> begin
            τ = sqrt(max(r2, zero_T))
            return s * coeff * exp(-τ) * kernel.polynomial(T(τ))
        end
    end
    return SignedStationaryKernelSpec(base_spec.scales, signed_map)
end

function stationary_kernel_spec(
        k::HalfIntegerMaternDerivativeEvenKernel,
        ::Type{T},
    ) where {T <: AbstractFloat}
    base_spec = stationary_kernel_spec(k.base, T)
    base_spec === nothing && return nothing
    coeff_T = T(k.coefficient)
    radial_map = let kernel = k, coeff = coeff_T, zero_T = zero(T)
        r2 -> begin
            τ = sqrt(max(r2, zero_T))
            value = coeff * exp(-τ) * kernel.polynomial(T(τ))
            return convert(T, value)
        end
    end
    return StationaryKernelSpec(base_spec.scales, radial_map)
end

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

function kernelmatrix(k::HalfIntegerMaternKernel, x::AbstractVector, y::AbstractVector)
    K = pairwise(k.dist, x, y)
    K .= _exp_poly.(Ref(k), K)
    return K
end

function radial_antiderivative(k::HalfIntegerMaternKernel{P}, ::Val{1}) where {P}
    sqrt_2nu = sqrt(2 * P + 1)
    neg_inv_sqrt_2nu = -(1.0 / sqrt_2nu)

    # Polynomial part: start with the base rational polynomial and add its successive derivatives
    # p_i ← base polynomial; poly ← p_i + p_i' + ... + p_i^(n)
    p_i = k.poly
    poly = p_i
    for _ in 1:P
        p_i = Polynomials.derivative(p_i)
        poly = poly + p_i
    end

    # C1 = -neg_inv_sqrt_2nu * constant_term(poly)
    C1 = -neg_inv_sqrt_2nu * coeffs(poly)[1]  # coeffs()[1] is c0 in Polynomials.jl

    # Return the radial antiderivative as a function of r
    return r -> neg_inv_sqrt_2nu * exp(-sqrt_2nu * r) * poly(sqrt_2nu * r) + C1
end

function radial_antiderivative(k::HalfIntegerMaternKernel{P}, ::Val{2}) where {P}
    sqrt_2nu = sqrt(2 * P + 1)
    inv_2nu = 1 / (2 * P + 1)
    neg_inv_sqrt_2nu = -(1 / sqrt_2nu)

    # Build both polynomials in one pass:
    # poly1 = p + p' + ... + p^(P)
    # poly2 = p + 2 p' + 3 p'' + ... + (P+1) p^(P)
    p_i = k.poly
    poly1 = p_i
    poly2 = p_i
    for i in 1:P
        p_i = Polynomials.derivative(p_i)
        poly1 += p_i
        poly2 += (i + 1) * p_i
    end

    # Constants
    C1 = -neg_inv_sqrt_2nu * coeffs(poly1)[1]   # from first antiderivative
    C2 = -inv_2nu * coeffs(poly2)[1]

    # Return the second radial antiderivative
    return r -> inv_2nu * exp(-sqrt_2nu * r) * poly2(sqrt_2nu * r) + C1 * r + C2
end

function derivative(
        k::HalfIntegerMaternKernel{P, ND},
        n::Int,
        m::Int,
    ) where {P, ND}
    n >= 0 || throw(ArgumentError("Derivative order must be non-negative"))
    m >= 0 || throw(ArgumentError("Derivative order must be non-negative"))
    ND == 1 || throw(ArgumentError("HalfIntegerMatern derivatives currently implemented only for 1D"))
    total = n + m
    if total == 0
        return k
    end
    poly = _derivative_polynomial(k, total)
    rho = _half_integer_matern_rho(k)
    coeff = (-1)^m * rho^total
    inner = if iseven(total)
        HalfIntegerMaternDerivativeEvenKernel(k, poly, coeff)
    else
        HalfIntegerMaternDerivativeOddKernel(k, poly, coeff, rho)
    end
    return DerivativeKernel1D{n, m}(k, inner)
end
