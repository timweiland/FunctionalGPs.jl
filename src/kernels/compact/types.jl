using Polynomials
import KernelFunctions
import Distances

export AbstractCompactKernel,
    AbstractCompactRadialKernel, AbstractCompactSignedRadialKernel
export CompactPolynomialKernel, CompactSignedPolynomialKernel

"""
    abstract type AbstractCompactKernel{X<:Number} <: KernelFunctions.Kernel
Abstract type representing a compact kernel function.

# Examples
```julia
julia> struct MyKernel <: AbstractCompactKernel{Float64} end

julia> lengthscales(k::MyKernel) = 0.5

julia> k_support(k::MyKernel, x, y) = sum(x .* y)
```
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
    abstract type AbstractCompactRadialKernel{X<:Number} <: AbstractCompactKernel{X}
Abstract type representing a compact radial kernel function.

# Examples
```julia
julia> struct MyRadialKernel <: AbstractCompactRadialKernel{Float64} end

julia> lengthscales(k::MyRadialKernel) = 0.1

julia> k_r(k::MyRadialKernel, r::Number) = exp(-r^2)
```
"""
abstract type AbstractCompactRadialKernel{X <: Number} <: AbstractCompactKernel{X} end

function k_r(k::AbstractCompactRadialKernel, r::Number)
    return error("k_r not implemented for $(typeof(k))")
end
function k_support(k::AbstractCompactRadialKernel, x, y)
    return (l = lengthscales(k); k_r(k, Distances.euclidean(x ./ l, y ./ l)))
end

"""
    abstract type AbstractCompactSignedRadialKernel{X<:Number} <: AbstractCompactKernel{X}
Abstract type representing a compact signed radial kernel function.

# Examples
```julia
julia> struct MySignedRadialKernel <: AbstractCompactSignedRadialKernel{Float64} end

julia> lengthscales(k::MySignedRadialKernel) = 0.1

julia> k_r(k::MySignedRadialKernel, r::Number) = exp(-r^2)
```
"""
abstract type AbstractCompactSignedRadialKernel{X <: Number} <: AbstractCompactKernel{X} end
function k_support(k::AbstractCompactSignedRadialKernel, x, y)
    return sign(x - y) * k_r(k, abs(x - y) / lengthscales(k))
end

"""
    struct CompactPolynomialKernel{T<:Number, X <: Number} <: AbstractCompactRadialKernel{X}

CompactPolynomialKernel represents a kernel function defined by a polynomial within its
    compact support.

# Fields
- `poly::Polynomial{T}`: The polynomial.
- `lengthscales::Union{X,AbstractVector{X}}`: The lengthscales, which also define the compact support.

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
    struct CompactSignedPolynomialKernel{T<:Number, X <: Number} <: AbstractCompactSignedRadialKernel{X}

CompactSignedPolynomialKernel represents a kernel function defined by a polynomial
    multiplied by the sign of x - y within its compact support.
For more information, see the documentation for `CompactPolynomialKernel`.

# Fields
- `poly::Polynomial{T}`: The polynomial used in the kernel.
- `lengthscales::Union{X,AbstractVector{X}}`: The lengthscales of the kernel.

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
