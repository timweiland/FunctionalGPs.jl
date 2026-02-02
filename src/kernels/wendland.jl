using Polynomials

export WendlandPolynomial, WendlandKernel

"""
    ϕ_l(l::Int) -> Polynomial

Compute the base truncated polynomial `ϕₗ(r) = (1-r)₊ˡ` in expanded form.

This is the building block for Wendland kernel construction. The polynomial
represents `max(1-r, 0)^l` expanded as a polynomial (valid for `r ∈ [0,1]`).

# Arguments
- `l`: Polynomial degree (positive integer)

# Returns
A `Polynomial` with rational coefficients representing the expansion of `(1-r)^l`.

Used internally by [`WendlandPolynomial`](@ref).
"""
pascal_row(n::Int) = [binomial(n, k) for k in 0:n]
ϕ_l(l::Int) = (bins = pascal_row(l); Polynomial(map(i -> (-1)^(i + 1) * bins[i], 1:(l + 1))))

"""
    WendlandPolynomial(d::Int, k::Int) -> Polynomial

Construct the polynomial defining a Wendland compactly-supported kernel.

Wendland kernels are a family of positive-definite radial basis functions
with minimal polynomial degree for a given smoothness. The resulting kernel
is `C^(2k)` smooth and positive definite in ℝᵈ.

# Arguments
- `d`: Spatial dimension (determines polynomial degree)
- `k`: Smoothness parameter (kernel is `C^(2k)` smooth)

# Returns
A `Polynomial` with rational coefficients, normalized so `p(0) = 1`.

# Examples
```julia
# Polynomial for Wendland kernel in 1D with C² smoothness
p = WendlandPolynomial(1, 1)

# Polynomial for Wendland kernel in 3D with C⁴ smoothness
p = WendlandPolynomial(3, 2)
```

See also: [`WendlandKernel`](@ref)
"""
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

"""
    WendlandKernel(d::Int, k::Int, [lengthscales])

Construct a Wendland compactly-supported kernel.

Wendland kernels are positive-definite, have compact support (exactly zero
outside a radius), and tunable smoothness. They enable sparse kernel matrices
for large-scale problems.

The kernel is defined as:
```
k(x, y) = p(‖x - y‖/ℓ)  if ‖x - y‖/ℓ ≤ 1
k(x, y) = 0              otherwise
```
where `p` is the Wendland polynomial and `ℓ` is the lengthscale.

# Arguments
- `d`: Spatial dimension (positive integer)
- `k`: Smoothness parameter (non-negative integer). The kernel is `C^(2k)` smooth.
- `lengthscales`: (Optional) Scalar or vector of lengthscales. Default is `1.0`.

# Common Choices
| d | k | Smoothness | Use Case                    |
|---|---|------------|-----------------------------|
| 1 | 1 | C²         | 1D regression               |
| 1 | 2 | C⁴         | 1D with derivative data     |
| 3 | 1 | C²         | 3D spatial interpolation    |
| 3 | 2 | C⁴         | 3D with smoothness needs    |

# Examples
```julia
# Basic Wendland kernel in 1D with C⁴ smoothness
k = WendlandKernel(1, 2)
k([0.0], [0.5])  # Evaluate kernel

# With custom lengthscale
k = WendlandKernel(1, 2, 0.3)

# In 2D with per-dimension lengthscales
k = WendlandKernel(2, 1, [0.5, 1.0])
```

See also: [`WendlandPolynomial`](@ref), [`CompactPolynomialKernel`](@ref), [`HalfIntegerMaternKernel`](@ref)
"""
WendlandKernel(d::Int, k::Int) = CompactPolynomialKernel(WendlandPolynomial(d, k), 1.0)
function WendlandKernel(
        d::Int,
        k::Int,
        lengthscales::Union{Number, AbstractVector{Number}},
    )
    return CompactPolynomialKernel(WendlandPolynomial(d, k), lengthscales)
end
