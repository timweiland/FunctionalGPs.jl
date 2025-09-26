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

function _exp_poly(k::HalfIntegerMaternKernel, d)
    return exp(-d) * k.poly(d)
end

(k::HalfIntegerMaternKernel)(x, y) = _exp_poly(k, k.dist(x, y))

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
