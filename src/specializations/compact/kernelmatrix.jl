import KernelFunctions: kernelmatrix, ColVecs, RowVecs
import NearestNeighbors: BallTree, inrange, NNTree
import SparseArrays.sparse

"""
    SearchTree(x, metric)

Construct a `BallTree` spatial index for efficient neighbor queries. Handles
various input formats (`AbstractVector`, `RowVecs`, `ColVecs`, numeric vectors)
by converting to the matrix layout expected by `NearestNeighbors.jl`.
"""
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

"""
    inrange_point(tree, point, radius)

Find all points in `tree` within `radius` of `point`. Wraps scalar points into
a vector for compatibility with `NearestNeighbors.inrange`.
"""
function inrange_point(tree::NNTree, point::T, radius::Number) where {T <: Number}
    return inrange(tree, [point], radius)
end
inrange_point(tree::NNTree, point, radius::Number) = inrange(tree, point, radius)

"""
    kernelmatrix(k::AbstractCompactKernel, x, y)
    kernelmatrix(k::AbstractCompactKernel, x)

Construct a sparse covariance matrix for a compactly supported kernel. Uses a
`BallTree` to efficiently find pairs of points within the kernel's support
radius, avoiding O(n²) comparisons.

Returns a `SparseMatrixCSC` with non-zero entries only where `k(x[i], y[j]) ≠ 0`.
"""
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
