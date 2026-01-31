export derivative

"""
    derivative(k::CompactPolynomialKernel, n::Int, m::Int)

Compute the derivative of a CompactPolynomialKernel.

# Arguments
- `k::CompactPolynomialKernel`: The CompactPolynomialKernel object.
- `n::Int`: Order along the first argument.
- `m::Int`: Order along the second argument.

# Returns
- `CompactPolynomialKernel` or `CompactSignedPolynomialKernel`: The derivative of the
CompactPolynomialKernel object.

"""
function derivative(k::CompactPolynomialKernel, n::Int, m::Int)
    if n == 0 && m == 0
        return k
    end
    total = n + m
    poly = (-1)^m * Polynomials.derivative(k.poly, n + m) / (k.lengthscales^total)
    inner_kernel =
        iseven(total) ? CompactPolynomialKernel(poly, k.lengthscales) :
        CompactSignedPolynomialKernel(poly, k.lengthscales)
    return DerivativeKernel1D{n, m}(k, inner_kernel)
end

function derivative(k::CompactSignedPolynomialKernel, n::Int, m::Int)
    if n == 0 && m == 0
        return k
    end
    total = n + m
    poly = (-1)^m * Polynomials.derivative(k.poly, n + m) / (k.lengthscales^total)
    inner_kernel =
        isodd(total) ? CompactPolynomialKernel(poly, k.lengthscales) :
        CompactSignedPolynomialKernel(poly, k.lengthscales)
    return DerivativeKernel1D{n, m}(k, inner_kernel)
end
