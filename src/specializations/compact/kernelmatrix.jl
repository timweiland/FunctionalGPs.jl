import KernelFunctions: kernelmatrix, ColVecs, RowVecs
import NearestNeighbors: BallTree, inrange, NNTree
import SparseArrays.sparse

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
