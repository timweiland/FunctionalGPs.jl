export SignedStationaryKernelMatrix

import LinearAlgebra
using .StationaryUtils: _scaled_inputs_and_norms, _stationary_scale_values

"""
    SignedStationaryKernelMatrix{T,TX,TY,TV,TW,F}

Lazy stationary matrix where the kernel depends on both squared distance and
signed differences between points. Suitable for one-sided integrals and odd
derivatives that introduce direction-dependent factors.
"""
struct SignedStationaryKernelMatrix{T, TX <: AbstractMatrix{T}, TY <: AbstractMatrix{T}, TV <: AbstractVector{T}, TW <: AbstractVector{T}, F} <: CovarianceMatrix{T}
    scaled_left::TX
    scaled_right::TY
    norms_left::TV
    norms_right::TW
    signed_kernel_map::F
end

"""
    SignedStationaryKernelMatrix(X, ϕ; scales = nothing)

Construct a signed stationary matrix using the same inputs on both sides.
"""
function SignedStationaryKernelMatrix(
        X::AbstractMatrix,
        ϕ::F;
        scales::Union{Nothing, AbstractVector} = nothing,
    ) where {F}
    return SignedStationaryKernelMatrix(X, X, ϕ; scales = scales)
end

"""
    SignedStationaryKernelMatrix(X_left, X_right, ϕ; scales = nothing)

Create a signed stationary matrix evaluating `ϕ(r2, Δ)` where `r2` is the
squared distance between scaled points and `Δ` is the signed difference in the
first (and only) dimension.
"""
function SignedStationaryKernelMatrix(
        X_left::AbstractMatrix,
        X_right::AbstractMatrix,
        ϕ::F;
        scales::Union{Nothing, AbstractVector} = nothing,
    ) where {F}
    size(X_left, 2) == size(X_right, 2) ||
        throw(DimensionMismatch("point sets must share feature dimension"))
    ncols = size(X_left, 2)
    ncols == 1 ||
        throw(ArgumentError("Signed stationary matrices currently support only 1D inputs"))
    scale_values = _stationary_scale_values(ncols, scales)
    scaled_left, norms_left = _scaled_inputs_and_norms(X_left, scale_values)
    if X_left === X_right
        scaled_right = scaled_left
        norms_right = norms_left
    else
        scaled_right, norms_right = _scaled_inputs_and_norms(X_right, scale_values)
    end
    return SignedStationaryKernelMatrix(
        scaled_left,
        scaled_right,
        norms_left,
        norms_right,
        ϕ,
    )
end

"""
    SignedStationaryKernelMatrix(scaled_left, scaled_right, norms_left, norms_right, ϕ)

Internal constructor when scaled representations are already available.
"""
function SignedStationaryKernelMatrix(
        scaled_left::TX,
        scaled_right::TY,
        norms_left::TV,
        norms_right::TW,
        ϕ::F,
    ) where {T <: Real, TX <: AbstractMatrix{T}, TY <: AbstractMatrix{T}, TV <: AbstractVector{T}, TW <: AbstractVector{T}, F}
    return SignedStationaryKernelMatrix{T, TX, TY, TV, TW, F}(
        scaled_left,
        scaled_right,
        norms_left,
        norms_right,
        ϕ,
    )
end

function LinearAlgebra.mul!(
        y::AbstractVector{T},
        K::SignedStationaryKernelMatrix{T},
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
                    Δ = sign(Xi[ii, 1] - Xj[jj, 1])
                    tmp[ii] += K.signed_kernel_map(r2, Δ) * vjj
                end
            end
            @views y[I] .+= tmp
        end
    end
    return y
end

function Base.:*(K::SignedStationaryKernelMatrix{T}, v::AbstractVector{T}) where {T <: Real}
    y = similar(v, T, size(K, 1))
    return LinearAlgebra.mul!(y, K, v)
end

function Base.size(K::SignedStationaryKernelMatrix)
    return (size(K.scaled_left, 1), size(K.scaled_right, 1))
end

function Base.getindex(K::SignedStationaryKernelMatrix{T}, i::Int, j::Int) where {T}
    Xi = @view K.scaled_left[i, :]
    Xj = @view K.scaled_right[j, :]
    d2 = K.norms_left[i] + K.norms_right[j] - 2 * dot(Xi, Xj)
    d2 = ifelse(d2 < zero(T), zero(T), d2)
    Δ = sign(K.scaled_left[i, 1] - K.scaled_right[j, 1])
    return K.signed_kernel_map(d2, Δ)
end

function Base.getindex(
        K::SignedStationaryKernelMatrix{T},
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
    Δ = sign.(Xi[:, 1] .- (Xj[:, 1])')
    return K.signed_kernel_map.(gram, Δ)
end

function Base.Matrix(K::SignedStationaryKernelMatrix{T}) where {T <: Real}
    n_rows, n_cols = size(K)
    gram = similar(K.scaled_left, T, n_rows, n_cols)
    LinearAlgebra.mul!(gram, K.scaled_left, K.scaled_right', -2, zero(T))
    gram .+= K.norms_left .+ K.norms_right'
    Δ = sign.(K.scaled_left[:, 1] .- (K.scaled_right[:, 1])')
    return K.signed_kernel_map.(gram, Δ)
end
