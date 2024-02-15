using Polynomials

export WendlandPolynomial, WendlandKernel, ϕ_l

pascal_row(n::Int) = [binomial(n, k) for k in 0:n]
ϕ_l(l::Int) = (bins = pascal_row(l); Polynomial(map(i -> (-1)^(i + 1) * bins[i], 1:l+1)))

function WendlandPolynomial(d::Int, k::Int)
    l = Int(d ÷ 2 + k + 1)
    ϕ = ϕ_l(l)
    coeffs = ϕ.coeffs
    for _ in 1:k
        coeff_0 = sum([c // (j + 1) for (j, c) in enumerate(coeffs)])
        coeffs_rest = [-coeffs[j] // (j + 1) for j in eachindex(coeffs)]
        coeffs = [coeff_0, 0, coeffs_rest...]
    end
    coeffs //= coeffs[1]
    return Polynomial(coeffs)
end

WendlandKernel(d::Int, k::Int) = CompactPolynomialKernel(WendlandPolynomial(d, k), 1.0)
function WendlandKernel(
    d::Int,
    k::Int,
    lengthscales::Union{Number,AbstractVector{Number}},
)
    return CompactPolynomialKernel(WendlandPolynomial(d, k), lengthscales)
end
