using Polynomials
import KernelFunctions
import Distances
import KernelFunctions: kernelmatrix, ColVecs, RowVecs
import NearestNeighbors: BallTree, inrange, NNTree

export AbstractCompactKernel, AbstractCompactRadialKernel, AbstractCompactSignedRadialKernel
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
abstract type AbstractCompactKernel{X<:Number} <: KernelFunctions.Kernel end

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

SearchTree(x::AbstractVector, metric::Distances.UnionMetric) = BallTree(x, metric)
SearchTree(x::RowVecs, metric::Distances.UnionMetric) = BallTree(x.X', metric)
SearchTree(x::ColVecs, metric::Distances.UnionMetric) = BallTree(x.X, metric)
function SearchTree(x::AbstractVector{T}, metric::Distances.UnionMetric) where {T<:Number}
    return BallTree(convert(Matrix{Float64}, reshape(x, 1, :)), metric)
end
inrange_point(tree::NNTree, point::T, radius::Number) where {T<:Number} = inrange(tree, [point], radius)
inrange_point(tree::NNTree, point, radius::Number) = inrange(tree, point, radius)

function kernelmatrix(k::AbstractCompactKernel, x::AbstractVector, y::AbstractVector)
    max_dist = max(lengthscales(k)...)
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
    return sparse(I, J, V, length(x), length(y))
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
abstract type AbstractCompactRadialKernel{X<:Number} <: AbstractCompactKernel{X} end

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
abstract type AbstractCompactSignedRadialKernel{X<:Number} <: AbstractCompactKernel{X} end
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
struct CompactPolynomialKernel{T<:Number,X<:Number} <: AbstractCompactRadialKernel{X}
    poly::Polynomial{T}
    lengthscales::Union{X,AbstractVector{X}}
end

function CompactPolynomialKernel(poly::Polynomial{T}) where {T<:Number}
    return CompactPolynomialKernel{T}(poly, 1.0)
end
lengthscales(k::CompactPolynomialKernel) = k.lengthscales
k_r(k::CompactPolynomialKernel, r::Number) = k.poly(r)

"""
    struct CompactSignedPolynomialKernel{T<:Number, X <: Number} <: AbstractCompactSignedRadialKernel{X}

CompactSignedPolynomialKernel represents a kernel function defined by a polynomial 
    multiplied by the sign of x - y within its compact support.
For more information, see the documentation for `CompactPolynomialKernel`.

# Fields
- `poly::Polynomial{T}`: The polynomial used in the kernel.
- `lengthscales::Union{X,AbstractVector{X}}`: The lengthscales of the kernel.

"""
struct CompactSignedPolynomialKernel{T<:Number,X<:Number} <:
       AbstractCompactSignedRadialKernel{X}
    poly::Polynomial{T}
    lengthscales::Union{X,AbstractVector{X}}
end

function CompactSignedPolynomialKernel(poly::Polynomial{T}) where {T<:Number}
    return CompactSignedPolynomialKernel{T}(poly, 1.0)
end
lengthscales(k::CompactSignedPolynomialKernel) = k.lengthscales
k_r(k::CompactSignedPolynomialKernel, r::Number) = k.poly(r)

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
    total = n + m
    poly = (-1)^m * Polynomials.derivative(k.poly, n + m) / (k.lengthscales^total)
    return iseven(total) ? CompactPolynomialKernel(poly, k.lengthscales) :
           CompactSignedPolynomialKernel(poly, k.lengthscales)
end
