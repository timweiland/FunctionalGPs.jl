using Polynomials
import KernelFunctions
import Distances

export AbstractCompactKernel,
    AbstractCompactRadialKernel, AbstractCompactSignedRadialKernel
export CompactPolynomialKernel, CompactSignedPolynomialKernel

"""
    AbstractCompactKernel{X} <: Kernel

Abstract base type for compactly-supported kernels.

Compact kernels are exactly zero when inputs are farther apart than a
threshold (determined by `lengthscales`). This enables sparse kernel matrices
and efficient computation for large datasets.

# Interface
Subtypes must implement:
- `lengthscales(k)`: Return the lengthscale(s) defining the support radius
- `k_support(k, x, y)`: Evaluate the kernel within its support region

# Type Parameter
- `X`: Element type of the lengthscale parameter

See also: [`AbstractCompactRadialKernel`](@ref), [`CompactPolynomialKernel`](@ref)
"""
abstract type AbstractCompactKernel{X <: Number} <: KernelFunctions.Kernel end

function lengthscales(k::AbstractCompactKernel)
    return error("lengthscales not implemented for $(typeof(k))")
end
function k_support(k::AbstractCompactKernel, x, y)
    return error("k_support not implemented for $(typeof(k))")
end

function (k::AbstractCompactKernel)(x, y)
    r = Distances.euclidean(x ./ lengthscales(k), y ./ lengthscales(k))
    return r <= 1 ? k_support(k, x, y) : 0.0
end

"""
    AbstractCompactRadialKernel{X} <: AbstractCompactKernel{X}

Abstract type for compact radial (isotropic) kernels.

Radial kernels depend only on the distance `r = ‖x - y‖/ℓ` between inputs.
Combined with compact support, these kernels enable both sparsity and
efficient stationary-structure optimizations.

# Interface
Subtypes must implement:
- `lengthscales(k)`: Return the lengthscale(s)
- `k_r(k, r)`: Evaluate the kernel as a function of normalized distance `r ∈ [0,1]`

The `k_support` method is automatically defined via `k_r`.

See also: [`AbstractCompactKernel`](@ref), [`CompactPolynomialKernel`](@ref)
"""
abstract type AbstractCompactRadialKernel{X <: Number} <: AbstractCompactKernel{X} end

function k_r(k::AbstractCompactRadialKernel, r::Number)
    return error("k_r not implemented for $(typeof(k))")
end
function k_support(k::AbstractCompactRadialKernel, x, y)
    return (l = lengthscales(k); k_r(k, Distances.euclidean(x ./ l, y ./ l)))
end

"""
    AbstractCompactSignedRadialKernel{X} <: AbstractCompactKernel{X}

Abstract type for compact signed radial kernels.

Signed radial kernels have the form `k(x,y) = sign(x-y) · f(|x-y|/ℓ)`.
This structure arises from odd-order derivatives of radial kernels.
Primarily used in 1D for derivative kernel representations.

# Interface
Subtypes must implement:
- `lengthscales(k)`: Return the lengthscale
- `k_r(k, r)`: Evaluate the unsigned radial factor for `r ∈ [0,1]`

The `k_support` method is automatically defined with the sign factor.

See also: [`AbstractCompactKernel`](@ref), [`CompactSignedPolynomialKernel`](@ref)
"""
abstract type AbstractCompactSignedRadialKernel{X <: Number} <: AbstractCompactKernel{X} end
function k_support(k::AbstractCompactSignedRadialKernel, x, y)
    return sign(x - y) * k_r(k, abs(x - y) / lengthscales(k))
end

"""
    CompactPolynomialKernel{T, X} <: AbstractCompactRadialKernel{X}

A compactly-supported kernel defined by evaluating a polynomial on normalized distance.

The kernel is:
```
k(x, y) = poly(r)  if r = ‖x - y‖/ℓ ≤ 1
k(x, y) = 0        otherwise
```
where `poly` is the defining polynomial and `ℓ` is the lengthscale.

# Type Parameters
- `T`: Coefficient type of the polynomial
- `X`: Element type of the lengthscale

# Fields
- `poly`: Polynomial evaluated at normalized distance `r ∈ [0,1]`
- `lengthscales`: Scalar or vector defining the support radius

# Constructor
    CompactPolynomialKernel(poly::Polynomial, [lengthscales=1.0])

# Examples
```julia
using Polynomials

# Custom polynomial kernel: k(r) = 1 - r² for r ≤ 1
p = Polynomial([1, 0, -1])
k = CompactPolynomialKernel(p)

# With lengthscale 0.5 (support radius = 0.5)
k = CompactPolynomialKernel(p, 0.5)

# Wendland kernels are CompactPolynomialKernels
k = WendlandKernel(1, 2)  # Returns a CompactPolynomialKernel
```

See also: [`WendlandKernel`](@ref), [`CompactSignedPolynomialKernel`](@ref)
"""
struct CompactPolynomialKernel{T <: Number, X <: Number} <: AbstractCompactRadialKernel{X}
    poly::Polynomial{T}
    lengthscales::Union{X, AbstractVector{X}}
end

function CompactPolynomialKernel(poly::Polynomial{T}) where {T <: Number}
    return CompactPolynomialKernel{T}(poly, 1.0)
end
lengthscales(k::CompactPolynomialKernel) = k.lengthscales
k_r(k::CompactPolynomialKernel, r::Number) = k.poly(r)
kernel_structure(::CompactPolynomialKernel) = StationaryKernelTrait()

function Base.:(==)(k1::CompactPolynomialKernel, k2::CompactPolynomialKernel)
    return k1.poly == k2.poly && k1.lengthscales == k2.lengthscales
end
function Base.isapprox(k1::CompactPolynomialKernel, k2::CompactPolynomialKernel)
    return k1.poly ≈ k2.poly && k1.lengthscales ≈ k2.lengthscales
end

"""
    CompactSignedPolynomialKernel{T, X} <: AbstractCompactSignedRadialKernel{X}

A compactly-supported signed kernel defined by a polynomial times `sign(x-y)`.

The kernel is:
```
k(x, y) = sign(x - y) · poly(|x - y|/ℓ)  if |x - y|/ℓ ≤ 1
k(x, y) = 0                               otherwise
```

This structure arises from odd-order derivatives of compact polynomial kernels.
It is antisymmetric: `k(x, y) = -k(y, x)`.

# Type Parameters
- `T`: Coefficient type of the polynomial
- `X`: Element type of the lengthscale

# Fields
- `poly`: Polynomial evaluated at normalized distance
- `lengthscales`: Scalar or vector defining the support radius

# Constructor
    CompactSignedPolynomialKernel(poly::Polynomial, [lengthscales=1.0])

Typically constructed via [`derivative`](@ref) on a [`CompactPolynomialKernel`](@ref)
rather than directly.

See also: [`CompactPolynomialKernel`](@ref), [`derivative`](@ref)
"""
struct CompactSignedPolynomialKernel{T <: Number, X <: Number} <:
    AbstractCompactSignedRadialKernel{X}
    poly::Polynomial{T}
    lengthscales::Union{X, AbstractVector{X}}
end

function CompactSignedPolynomialKernel(poly::Polynomial{T}) where {T <: Number}
    return CompactSignedPolynomialKernel{T}(poly, 1.0)
end
lengthscales(k::CompactSignedPolynomialKernel) = k.lengthscales
k_r(k::CompactSignedPolynomialKernel, r::Number) = k.poly(r)

function Base.:(==)(k1::CompactSignedPolynomialKernel, k2::CompactSignedPolynomialKernel)
    return k1.poly == k2.poly && k1.lengthscales == k2.lengthscales
end
function Base.isapprox(
        k1::CompactSignedPolynomialKernel,
        k2::CompactSignedPolynomialKernel,
    )
    return k1.poly ≈ k2.poly && k1.lengthscales ≈ k2.lengthscales
end
