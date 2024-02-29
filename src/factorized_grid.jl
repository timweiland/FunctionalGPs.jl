import KernelFunctions: KernelTensorProduct, kernelmatrix, kernelmatrix_diag

export FactorizedGrid, kernelmatrix, kernelmatrix_diag

struct FactorizedGrid{T} <: AbstractVector{T}
    ranges::Tuple{Vararg{AbstractVector{T}}}
end

function Base.convert(::Type{Array{T}}, A::FactorizedGrid) where {T<:Number}
    X = collect(Iterators.product(A.ranges...)) |> (M -> reinterpret(reshape, T, M))
    return moveaxis(X, 1, ndims(X))
end

Base.convert(::Type{Array}, A::FactorizedGrid) = convert(Array{Float64}, A)

FactorizedGrid(ranges::AbstractVector{T}...) where {T} = FactorizedGrid{T}(ranges)

Base.size(A::FactorizedGrid) = Tuple(length(rangeᵢ) for rangeᵢ in A.ranges)

Base.size(A::FactorizedGrid, i::Integer) = length(A.ranges[i])

Base.getindex(A::FactorizedGrid, i::Integer) = A.ranges[i]

Base.axes(A::FactorizedGrid, i::Integer) = 1:length(A.ranges[i])

# Index into the grid
function Base.getindex(A::FactorizedGrid, I::Vararg{Integer,N}) where {N}
    @assert length(I) == length(A.ranges)
    return [A.ranges[i][I[i]] for i in 1:N]
end

function kernelmatrix(k::KernelTensorProduct, x::FactorizedGrid, y::FactorizedGrid)
    @assert length(x.ranges) == length(y.ranges)
    @assert length(k.kernels) == length(x.ranges)

    Ks =
        (kernelmatrix(k.kernels[i], x.ranges[i], y.ranges[i]) for i in 1:length(x.ranges))
    return reduce(kron, reverse(Tuple(Ks)))
end
kernelmatrix(k::KernelTensorProduct, x::FactorizedGrid) = kernelmatrix(k, x, x)
function kernelmatrix_diag(k::KernelTensorProduct, x::FactorizedGrid)
    @assert length(k.kernels) == length(x.ranges)
    diags = (kernelmatrix_diag(k.kernels[i], x.ranges[i]) for i in 1:length(x.ranges))
    return reduce(kron, diags)
end

function Base.show(io::IO, A::FactorizedGrid)
    print(io, "FactorizedGrid($(size(A)))")
end