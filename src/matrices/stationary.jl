export StationaryKernelMatrix

import LinearAlgebra

"""
    StationaryKernelMatrix{T,TX,TY,TV,TW,F}

Lazy stationary covariance or cross-covariance matrix. Fields:
- `scaled_left`: column-scaled left inputs of size `(n_left, d)`
- `scaled_right`: column-scaled right inputs of size `(n_right, d)`
- `norms_left`: row-wise squared norms of the left inputs
- `norms_right`: row-wise squared norms of the right inputs
- `kernel_map`: callable taking squared distance and returning covariance value
- `jitter`: diagonal adjustment applied lazily when both sides alias
"""
struct StationaryKernelMatrix{T, TX <: AbstractMatrix{T}, TY <: AbstractMatrix{T}, TV <: AbstractVector{T}, TW <: AbstractVector{T}, F} <:
    CovarianceMatrix{T}
    scaled_left::TX
    scaled_right::TY
    norms_left::TV
    norms_right::TW
    kernel_map::F
    jitter::T
end

using .StationaryUtils: _scaled_inputs_and_norms, _stationary_scale_values

"""
    StationaryKernelMatrix(X, ϕ; scales = nothing, jitter = zero(T))

Construct a stationary covariance matrix using the same set of inputs on both
sides.
"""
function StationaryKernelMatrix(
        X::AbstractMatrix{T},
        ϕ::F;
        scales::Union{Nothing, AbstractVector} = nothing,
        jitter = zero(T),
    ) where {T <: Real, F}
    return StationaryKernelMatrix(X, X, ϕ; scales = scales, jitter = jitter)
end

"""
    StationaryKernelMatrix(X_left, X_right, ϕ; scales = nothing, jitter = zero(T))

Create a stationary cross-covariance matrix between the point sets `X_left` and
`X_right`. Both inputs must admit the same number of columns. Optional `scales`
specify per-dimension scalings applied to both sets; `jitter` is only applied
when `X_left === X_right`.
"""
function StationaryKernelMatrix(
        X_left::AbstractMatrix,
        X_right::AbstractMatrix,
        ϕ::F;
        scales::Union{Nothing, AbstractVector} = nothing,
        jitter = zero(eltype(X_left)),
    ) where {F}
    size(X_left, 2) == size(X_right, 2) ||
        throw(DimensionMismatch("point sets must share feature dimension"))
    ncols = size(X_left, 2)
    scale_values = _stationary_scale_values(ncols, scales)
    scaled_left, norms_left = _scaled_inputs_and_norms(X_left, scale_values)
    if X_left === X_right
        scaled_right = scaled_left
        norms_right = norms_left
    else
        scaled_right, norms_right = _scaled_inputs_and_norms(X_right, scale_values)
    end
    # Promote jitter to match the element type of scaled data
    T_scaled = eltype(scaled_left)
    return StationaryKernelMatrix(
        scaled_left,
        scaled_right,
        norms_left,
        norms_right,
        ϕ,
        T_scaled(jitter),
    )
end

"""
    StationaryKernelMatrix(scaled_left, scaled_right, norms_left, norms_right, ϕ, jitter)

Internal constructor used when scaled inputs and norms are already available.
"""
function StationaryKernelMatrix(
        scaled_left::TX,
        scaled_right::TY,
        norms_left::TV,
        norms_right::TW,
        ϕ::F,
        jitter::T,
    ) where {T <: Real, TX <: AbstractMatrix{T}, TY <: AbstractMatrix{T}, TV <: AbstractVector{T}, TW <: AbstractVector{T}, F}
    return StationaryKernelMatrix{T, TX, TY, TV, TW, F}(
        scaled_left,
        scaled_right,
        norms_left,
        norms_right,
        ϕ,
        jitter,
    )
end


"""
    LinearAlgebra.mul!(y, K::StationaryKernelMatrix, v; iblock = 1024, jblock = 8192)

Compute `y .= K * v` without materialising `K`, using tiled batched matrix
products so that GPU backends can dispatch to their BLAS implementations. The
optional block sizes tune workspace usage.
"""
function LinearAlgebra.mul!(
        y::AbstractVector{T},
        K::StationaryKernelMatrix{T},
        v::AbstractVector{T};
        iblock::Integer = 1024,
        jblock::Integer = 8192,
    ) where {T <: Real}
    n_rows = size(K, 1)
    n_cols = size(K, 2)
    if length(y) != n_rows || length(v) != n_cols
        throw(DimensionMismatch("Vector lengths must match matrix dimensions"))
    end
    fill!(y, zero(T))

    for i1 in 1:iblock:n_rows
        i2 = min(i1 + iblock - 1, n_rows)
        I = i1:i2
        Xi = @view K.scaled_left[I, :]
        ni = @view K.norms_left[I]

        for j1 in 1:jblock:n_cols
            j2 = min(j1 + jblock - 1, n_cols)
            J = j1:j2
            Xj = @view K.scaled_right[J, :]
            nj = @view K.norms_right[J]
            vJ = @view v[J]

            gram = similar(K.scaled_left, T, length(I), length(J))
            LinearAlgebra.mul!(gram, Xi, Xj', one(T), zero(T))

            tmp = similar(K.norms_left, T, length(I))
            fill!(tmp, zero(T))

            @inbounds for (jj, j) in enumerate(J)
                njj = nj[jj]
                vjj = vJ[jj]
                @inbounds for ii in 1:length(I)
                    r2 = ni[ii] + njj - 2 * gram[ii, jj]
                    r2 = ifelse(r2 < zero(T), zero(T), r2)
                    tmp[ii] += K.kernel_map(r2) * vjj
                end
            end
            @views y[I] .+= tmp
        end
    end

    if K.jitter != zero(T) && K.scaled_left === K.scaled_right
        @inbounds @simd for i in 1:n_rows
            y[i] += K.jitter * v[i]
        end
    end
    return y
end

"""
    *(K::StationaryKernelMatrix, v)

Allocate the result of the matrix–vector product `K * v` using the lazy
`tiled` multiplication defined in `mul!`.
"""
function Base.:*(K::StationaryKernelMatrix{T}, v::AbstractVector{T}) where {T <: Real}
    y = similar(v, T, size(K, 1))
    return LinearAlgebra.mul!(y, K, v)
end

"""
    size(K::StationaryKernelMatrix)

Return the matrix dimensions. Stationary kernel matrices are square, so the
dimensions share the same length.
"""
function Base.size(K::StationaryKernelMatrix)
    return (size(K.scaled_left, 1), size(K.scaled_right, 1))
end

"""
    getindex(K::StationaryKernelMatrix, i, j)

Return the `(i, j)` entry without materialising the full matrix. The kernel is
evaluated on the rescaled inputs and the diagonal jitter is applied when
`i == j`.
"""
function Base.getindex(K::StationaryKernelMatrix{T}, i::Int, j::Int) where {T}
    Xi = @view K.scaled_left[i, :]
    Xj = @view K.scaled_right[j, :]
    d2 = K.norms_left[i] + K.norms_right[j] - 2 * dot(Xi, Xj)
    d2 = ifelse(d2 < zero(T), zero(T), d2)
    value = K.kernel_map(d2)
    if K.jitter != zero(T) && K.scaled_left === K.scaled_right && i == j
        return value + K.jitter
    end
    return value
end

"""
    getindex(K::StationaryKernelMatrix, rows, cols)

Extract a rectangular block defined by integer index vectors `rows` and `cols`.
The method avoids materialising the full matrix by reusing block GEMM
operations on the scaled inputs.
"""
function Base.getindex(
        K::StationaryKernelMatrix{T},
        rows::AbstractVector{<:Integer},
        cols::AbstractVector{<:Integer},
    ) where {T}
    Xi = @view K.scaled_left[rows, :]
    Xj = @view K.scaled_right[cols, :]
    ni = @view K.norms_left[rows]
    nj = @view K.norms_right[cols]

    gram = similar(K.scaled_left, T, length(rows), length(cols))
    LinearAlgebra.mul!(gram, Xi, Xj', -2, zero(T))
    gram .+= ni .+ nj'
    block = K.kernel_map.(gram)

    if K.scaled_left === K.scaled_right && rows === cols && K.jitter != zero(T)
        diagindices = LinearAlgebra.diagind(block)
        block[diagindices] .+= K.jitter
    end
    return block
end

"""
    Matrix(K::StationaryKernelMatrix)

Materialise the full covariance matrix corresponding to `K`. The method keeps
all computation on the same device as `K` by leveraging GEMM calls.
"""
function Base.Matrix(K::StationaryKernelMatrix{T}) where {T <: Real}
    n_rows, n_cols = size(K)
    gram = similar(K.scaled_left, T, n_rows, n_cols)
    LinearAlgebra.mul!(gram, K.scaled_left, K.scaled_right', -2, zero(T))
    gram .+= K.norms_left .+ K.norms_right'
    values = K.kernel_map.(gram)

    if K.jitter != zero(T) && K.scaled_left === K.scaled_right
        diagindices = LinearAlgebra.diagind(values)
        values[diagindices] .+= K.jitter
    end
    return values
end
