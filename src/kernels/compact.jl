using Polynomials
import KernelFunctions
import Distances
import KernelFunctions: kernelmatrix, ColVecs, RowVecs
import NearestNeighbors: BallTree, inrange, NNTree
import SparseArrays.sparse

export AbstractCompactKernel,
    AbstractCompactRadialKernel, AbstractCompactSignedRadialKernel
export CompactPolynomialKernel, CompactSignedPolynomialKernel
export derivative

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

SearchTree(x::AbstractVector, metric::Distances.UnionMetrics) = BallTree(x, metric)
SearchTree(x::RowVecs, metric::Distances.UnionMetrics) = BallTree(x.X', metric)
SearchTree(x::ColVecs, metric::Distances.UnionMetrics) = BallTree(x.X, metric)
function SearchTree(
        x::AbstractVector{T},
        metric::Distances.UnionMetrics,
    ) where {T <: Number}
    # Ensure a 1×N floating matrix for BallTree while preserving values
    return BallTree(reshape(float.(collect(x)), 1, :), metric)
end
function inrange_point(tree::NNTree, point::T, radius::Number) where {T <: Number}
    return inrange(tree, [point], radius)
end
inrange_point(tree::NNTree, point, radius::Number) = inrange(tree, point, radius)

function kernelmatrix(k::AbstractCompactKernel, x::AbstractVector, y::AbstractVector)
    ls = lengthscales(k)
    max_dist = ls isa Number ? ls : maximum(ls)
    y_tree = SearchTree(y, Distances.euclidean)
    I, J = [], []
    V::Vector{Float64} = []
    for i in eachindex(x)
        neighbors = inrange_point(y_tree, x[i], max_dist + 0.001)
        for j in neighbors
            val = k(x[i], y[j])
            if val != 0.0
                push!(I, i)
                push!(J, j)
                push!(V, val)
            end
        end
    end
    return sparse(I, J, V, Base.length(x), Base.length(y))
end
kernelmatrix(k::AbstractCompactKernel, x::AbstractVector) = kernelmatrix(k, x, x)

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

function radial_antiderivative(k::CompactPolynomialKernel, ::Val{1})
    poly_int = Polynomials.integrate(k.poly)
    return (r) -> poly_int(min(r, 1.0))
end

function radial_antiderivative(k::CompactPolynomialKernel, ::Val{2})
    poly_int = Polynomials.integrate(k.poly)
    poly_int2 = Polynomials.integrate(poly_int)
    poly_int2_norm = poly_int2 - poly_int2.coeffs[1]
    return (r) -> (
        r > 1.0 ? poly_int2_norm(1.0) + (r - 1.0) * poly_int(1.0) : poly_int2_norm(r)
    )
end
