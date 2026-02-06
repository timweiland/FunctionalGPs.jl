import KernelFunctions: KernelTensorProduct, kernelmatrix, kernelmatrix_diag

export FactorizedGrid, kernelmatrix, kernelmatrix_diag

"""
    FactorizedGrid{T} <: AbstractVector{T}

A tensor product grid stored in factorized form for efficient Kronecker computations.

Rather than materializing an N-dimensional grid as a dense array of points, a
`FactorizedGrid` stores the 1D ranges for each dimension and computes products lazily.
This enables efficient kernel matrix computation via Kronecker products when used with
`KernelTensorProduct` kernels.

# Construction
Typically created via [`uniform_grid_n`](@ref) or [`uniform_grid_step`](@ref) on a
[`BoxDomain`](@ref), or directly from ranges:

```julia
grid = FactorizedGrid(0.0:0.1:1.0, 0.0:0.2:2.0)
```

# Examples
```julia
julia> box = BoxDomain((0.0, 1.0), (0.0, 1.0))
julia> grid = uniform_grid_n(box, 3, 3)
FactorizedGrid((3, 3))

julia> size(grid)
(3, 3)

julia> convert(Array, grid)  # materialize to dense array
3×3 Matrix{...}
```

See also: [`BoxDomain`](@ref), [`uniform_grid_n`](@ref), [`kernelmatrix`](@ref).
"""
struct FactorizedGrid{T} <: AbstractVector{T}
    ranges::Tuple{T, Vararg{T}}
end

Base.eltype(A::FactorizedGrid) = promote_type(map(eltype, A.ranges)...)

function Base.convert(::Type{Array{T}}, A::FactorizedGrid) where {T <: Number}
    X = collect(Iterators.product(A.ranges...)) |> (M -> reinterpret(reshape, T, M))
    return moveaxis(X, 1, ndims(X))
end

Base.convert(::Type{Array}, A::FactorizedGrid) = convert(Array{eltype(A)}, A)

FactorizedGrid(ranges::AbstractVector...) = FactorizedGrid(ranges)

Base.size(A::FactorizedGrid) = Tuple(length(rangeᵢ) for rangeᵢ in A.ranges)

Base.size(A::FactorizedGrid, i::Integer) = length(A.ranges[i])

Base.getindex(A::FactorizedGrid, i::Integer) = A.ranges[i]

Base.axes(A::FactorizedGrid, i::Integer) = 1:length(A.ranges[i])

# Index into the grid
function Base.getindex(A::FactorizedGrid, I::Vararg{Integer, N}) where {N}
    @assert length(I) == length(A.ranges)
    return [A.ranges[i][I[i]] for i in 1:N]
end

"""
    kernelmatrix(k::KernelTensorProduct, x::FactorizedGrid, y::FactorizedGrid)
    kernelmatrix(k::KernelTensorProduct, x::FactorizedGrid)

Compute the kernel matrix between two [`FactorizedGrid`](@ref)s using Kronecker products.

For a tensor product kernel `k = k₁ ⊗ k₂ ⊗ ...` and factorized grids, the kernel matrix
decomposes as a Kronecker product of per-dimension kernel matrices, enabling efficient
computation and storage.

Returns a `Kronecker` type from Kronecker.jl that supports efficient linear algebra.
"""
function kernelmatrix(k::KernelTensorProduct, x::FactorizedGrid, y::FactorizedGrid)
    @assert length(x.ranges) == length(y.ranges)
    @assert length(k.kernels) == length(x.ranges)

    Ks =
        (kernelmatrix(k.kernels[i], x.ranges[i], y.ranges[i]) for i in 1:length(x.ranges))
    return reduce(kronecker, reverse(Tuple(Ks)))
end
kernelmatrix(k::KernelTensorProduct, x::FactorizedGrid) = kernelmatrix(k, x, x)

# Mixed case: FactorizedGrid with ColVecs or other AbstractVector
# Convert FactorizedGrid to ColVecs for compatibility
# Allow _to_colvecs to handle FactorizedGrid (defined after array_ops.jl)
_to_colvecs(X::FactorizedGrid) = _grid_to_colvecs(X)

function _grid_to_colvecs(grid::FactorizedGrid)
    # convert(Array, grid) returns shape (n1, n2, ..., d) where d is dimension count
    arr = convert(Array, grid)
    d = length(grid.ranges)
    n_total = prod(size(grid))
    # Move last axis (d) to first position: (n1, n2, ..., d) -> (d, n1, n2, ...)
    arr_permuted = permutedims(arr, (ndims(arr), 1:(ndims(arr) - 1)...))
    # Reshape to (d, n_total) - column-major order gives same ordering as vector-of-vectors
    return ColVecs(reshape(arr_permuted, d, n_total))
end

function kernelmatrix(k::KernelTensorProduct, x::FactorizedGrid, y::ColVecs)
    return kernelmatrix(k, _grid_to_colvecs(x), y)
end

function kernelmatrix(k::KernelTensorProduct, x::ColVecs, y::FactorizedGrid)
    return kernelmatrix(k, x, _grid_to_colvecs(y))
end

"""
    kernelmatrix_diag(k::KernelTensorProduct, x::FactorizedGrid)

Compute the diagonal of the kernel matrix for a [`FactorizedGrid`](@ref).

Returns the Kronecker product of per-dimension diagonals as a vector.
"""
function kernelmatrix_diag(k::KernelTensorProduct, x::FactorizedGrid)
    @assert length(k.kernels) == length(x.ranges)
    diags = (kernelmatrix_diag(k.kernels[i], x.ranges[i]) for i in 1:length(x.ranges))
    return reduce(kron, diags)
end

function Base.show(io::IO, A::FactorizedGrid)
    return print(io, "FactorizedGrid($(size(A)))")
end
