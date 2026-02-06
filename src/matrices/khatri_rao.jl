import LinearAlgebra

export KhatriRaoMatrix

"""
    KhatriRaoMatrix{Axis, T, TF} <: CovarianceMatrix{T}

Lazy Khatri-Rao (face-splitting) product of factor matrices.

Stores only the factor matrices and computes products on demand without
materializing the full Khatri-Rao matrix.

# Type Parameters
- `Axis`: 1 for column-wise, 2 for row-wise
- `T`: Element type
- `TF`: Factor matrix type

# Column-wise (`Axis=1`)
Factors `A₁ (m₁×n), A₂ (m₂×n), ...` share column count `n`.
Result shape: `(∏mₖ × n)`. Column `j` is `kron(Aₙ[:,j], ..., A₁[:,j])`.

# Row-wise (`Axis=2`)
Factors `A₁ (m×n₁), A₂ (m×n₂), ...` share row count `m`.
Result shape: `(m × ∏nₖ)`. Row `i` is `kron(Aₙ[i,:], ..., A₁[i,:])`.
"""
struct KhatriRaoMatrix{Axis, T, TF <: AbstractMatrix{T}} <: CovarianceMatrix{T}
    factors::Vector{TF}
end

function KhatriRaoMatrix{Axis}(factors::Vector{TF}) where {Axis, T, TF <: AbstractMatrix{T}}
    return KhatriRaoMatrix{Axis, T, TF}(factors)
end

function Base.size(K::KhatriRaoMatrix{1})
    return (prod(size.(K.factors, 1)), size(K.factors[1], 2))
end

function Base.size(K::KhatriRaoMatrix{2})
    return (size(K.factors[1], 1), prod(size.(K.factors, 2)))
end

# Decompose a linear index into a multi-index across given dimensions (column-major)
function _multi_index(idx::Int, dims)
    idx -= 1
    indices = Vector{Int}(undef, length(dims))
    for k in 1:length(dims)
        indices[k] = idx % dims[k] + 1
        idx = idx ÷ dims[k]
    end
    return indices
end

function Base.getindex(K::KhatriRaoMatrix{1}, i::Int, j::Int)
    dims = [size(f, 1) for f in K.factors]
    idxs = _multi_index(i, dims)
    return prod(K.factors[k][idxs[k], j] for k in eachindex(K.factors))
end

function Base.getindex(K::KhatriRaoMatrix{2}, i::Int, j::Int)
    dims = [size(f, 2) for f in K.factors]
    idxs = _multi_index(j, dims)
    return prod(K.factors[k][i, idxs[k]] for k in eachindex(K.factors))
end

# ── Matrix materialization ──

function Base.Matrix(K::KhatriRaoMatrix{1})
    return _khatri_rao_columns(K.factors)
end

function Base.Matrix(K::KhatriRaoMatrix{2})
    return _khatri_rao_rows(K.factors)
end

function _khatri_rao_columns(factors::Vector)
    n_cols = size(factors[1], 2)
    result = factors[1]
    for i in 2:length(factors)
        F = factors[i]
        m1, m2 = size(result, 1), size(F, 1)
        new_result = similar(result, m1 * m2, n_cols)
        for j in 1:n_cols
            new_result[:, j] = kron(F[:, j], result[:, j])
        end
        result = new_result
    end
    return result
end

function _khatri_rao_rows(factors::Vector)
    n_rows = size(factors[1], 1)
    result = factors[1]
    for i in 2:length(factors)
        F = factors[i]
        n1, n2 = size(result, 2), size(F, 2)
        new_result = similar(result, n_rows, n1 * n2)
        for j in 1:n_rows
            new_result[j, :] = kron(F[j, :], result[j, :])
        end
        result = new_result
    end
    return result
end

# ── Matrix-vector product ──

function LinearAlgebra.mul!(
        y::AbstractVector{T},
        K::KhatriRaoMatrix{1, T},
        v::AbstractVector{T},
    ) where {T}
    N = length(K.factors)
    if N == 2
        return _mul_colwise_2!(y, K.factors[1], K.factors[2], v)
    else
        return _mul_colwise_general!(y, K.factors, v)
    end
end

function LinearAlgebra.mul!(
        y::AbstractVector{T},
        K::KhatriRaoMatrix{2, T},
        v::AbstractVector{T},
    ) where {T}
    N = length(K.factors)
    if N == 2
        return _mul_rowwise_2!(y, K.factors[1], K.factors[2], v)
    else
        return _mul_rowwise_general!(y, K.factors, v)
    end
end

# N=2 fast path, column-wise: y = vec((A .* v') * B')
function _mul_colwise_2!(y, A, B, v)
    # A is (m₁ × n), B is (m₂ × n), v is (n,)
    # Result: (m₁*m₂,) = vec(A * Diagonal(v) * B')
    y .= vec((A .* v') * B')
    return y
end

# N=2 fast path, row-wise: y = sum((A * V) .* B; dims=2)
function _mul_rowwise_2!(y, A, B, v)
    # A is (m × n₁), B is (m × n₂), v is (n₁*n₂,)
    n₁ = size(A, 2)
    n₂ = size(B, 2)
    V = reshape(v, n₁, n₂)
    y .= vec(sum((A * V) .* B; dims = 2))
    return y
end

# General N fallback, column-wise
function _mul_colwise_general!(y, factors, v)
    fill!(y, zero(eltype(y)))
    n_cols = size(factors[1], 2)
    for j in 1:n_cols
        col = factors[1][:, j]
        for k in 2:length(factors)
            col = kron(factors[k][:, j], col)
        end
        y .+= v[j] .* col
    end
    return y
end

# General N fallback, row-wise
function _mul_rowwise_general!(y, factors, v)
    fill!(y, zero(eltype(y)))
    m = size(factors[1], 1)
    dims = [size(f, 2) for f in factors]
    n_total = prod(dims)
    for j in 1:n_total
        idxs = _multi_index(j, dims)
        for i in 1:m
            val = prod(factors[k][i, idxs[k]] for k in eachindex(factors))
            y[i] += v[j] * val
        end
    end
    return y
end

# Allocating *
function Base.:*(K::KhatriRaoMatrix{Axis, T}, v::AbstractVector{T}) where {Axis, T}
    y = similar(v, T, size(K, 1))
    return LinearAlgebra.mul!(y, K, v)
end

# Matrix * Matrix: column-by-column
function Base.:*(K::KhatriRaoMatrix{Axis, T}, V::AbstractMatrix{T}) where {Axis, T}
    Y = similar(V, T, size(K, 1), size(V, 2))
    for col in 1:size(V, 2)
        LinearAlgebra.mul!(view(Y, :, col), K, V[:, col])
    end
    return Y
end
