"""
    _half_integer_matern_rho(k::HalfIntegerMaternKernel{P})

Compute the scaling factor `ρ = √(2ν+1) / ℓ` for a half-integer Matérn kernel,
where `ν = P + 1/2` and `ℓ` is the lengthscale.
"""
function _half_integer_matern_rho(k::HalfIntegerMaternKernel{P}) where {P}
    ℓ = only(k.lengthscales)
    return sqrt(float(2 * P + 1)) / float(ℓ)
end

"""
    _float_polynomial(poly)

Convert polynomial coefficients to floating point for numerical stability.
"""
function _float_polynomial(poly::Polynomial)
    return Polynomial(float.(coeffs(poly)))
end

"""
    _matern_derivative_polynomial(poly, order)

Compute the polynomial factor for a Matérn derivative by repeatedly applying
the recurrence `p ← p' - p`, which arises from differentiating `exp(-r) * p(r)`.
"""
function _matern_derivative_polynomial(poly::Polynomial, order::Int)
    result = poly
    for _ in 1:order
        result = Polynomials.derivative(result) - result
    end
    return result
end

"""
    _derivative_polynomial(k, total_order)

Get the derivative polynomial for a half-integer Matérn kernel at the given
total derivative order.
"""
function _derivative_polynomial(k::HalfIntegerMaternKernel, total_order::Int)
    float_poly = _float_polynomial(k.poly)
    return _matern_derivative_polynomial(float_poly, total_order)
end

"""
    derivative(k::HalfIntegerMaternKernel, n::Int, m::Int)

Compute the mixed partial derivative kernel `∂ⁿ⁺ᵐk/∂xⁿ∂yᵐ` for a 1D half-integer
Matérn kernel. Returns a `DerivativeKernel1D` wrapping either an even or odd
derivative kernel, depending on the parity of `n + m`.

# Arguments
- `k`: The base half-integer Matérn kernel
- `n`: Derivative order with respect to the first argument
- `m`: Derivative order with respect to the second argument

# Examples
```julia
k = HalfIntegerMaternKernel(2, [1.0])  # Matérn 5/2
dk = derivative(k, 1, 1)  # Second mixed derivative
```
"""
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
