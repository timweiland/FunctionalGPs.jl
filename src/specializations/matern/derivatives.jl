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
