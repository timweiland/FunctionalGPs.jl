module StationaryKernels

using LinearAlgebra

# ——————————————————————————————————————————————————————————
# StationaryKernelMatrix: K[i,j] = ϕ(‖X̃[i,:] - X̃[j,:]‖²) [+ jitter on diag]
# X̃ is the feature-scaled data: X̃[:,k] = X[:,k] * s[k]
# ϕ :: Function maps squared distance (r2::T) ↦ covariance (k::T).
# Works with Array{T,2} and CuArray{T,2} (if CUDA.jl is loaded).
# ——————————————————————————————————————————————————————————

struct StationaryKernelMatrix{T, TX, F} <: AbstractMatrix{T}
    X̃::TX                 # (n,d) scaled features; Array or CuArray
    nrm::AbstractVector{T} # (n,) rowwise ‖X̃‖², same device as X̃
    ϕ::F                   # callable r2 -> k
    jitter::T              # diagonal jitter (added on getindex/mul!)
end

Base.size(A::StationaryKernelMatrix) = (size(A.X̃, 1), size(A.X̃, 1))
Base.eltype(::Type{StationaryKernelMatrix{T}}) where {T} = T
Base.eltype(A::StationaryKernelMatrix{T}) where {T} = T

# Constructor: X (n,d), ϕ(d2)->k, optional per-dim scales s such that X̃ = X .* s'
# Pass e.g. s = 1 ./ ℓ for ARD (so d2 is Mahalanobis under ℓ)
function StationaryKernelMatrix(
        X::AbstractMatrix{T},
        ϕ::F;
        s::AbstractVector{T} = ones(T, size(X, 2)),
        jitter::T = convert(T, 0)
    ) where {T <: AbstractFloat, F}
    @assert size(s, 1) == size(X, 2)
    X̃ = similar(X)
    @. X̃ = X * s'                # broadcast column scaling; works on CPU & GPU
    nrm = sum(abs2, X̃; dims = 2)   # (n,1)
    nrm = vec(nrm)                # (n,)
    return StationaryKernelMatrix{T, typeof(X̃), F}(X̃, nrm, ϕ, jitter)
end

# O(1) access: ϕ(nrm[i] + nrm[j] - 2*dot(X̃[i], X̃[j])) (+ jitter on diag)
function Base.getindex(A::StationaryKernelMatrix{T}, i::Int, j::Int) where {T}
    Xi = @view A.X̃[i, :]
    Xj = @view A.X̃[j, :]
    d2 = A.nrm[i] + A.nrm[j] - 2 * dot(Xi, Xj)
    # Guard tiny negative due to roundoff:
    d2 = ifelse(d2 < zero(T), zero(T), d2)
    k = A.ϕ(d2)
    return (i == j) ? k + A.jitter : k
end

function Base.getindex(A::StationaryKernelMatrix{T}, Is::AbstractVector{Int}, Js::AbstractVector{Int}) where {T}
    Xi = @view A.X̃[Is, :]
    nrm_i = @view A.nrm[Is]
    if Is === Js
        Xj = Xi
        nrm_j = nrm_i
    else
        Xj = @view A.X̃[Js, :]
        nrm_j = @view A.nrm[Js]
    end

    K = similar(Xi, T, length(Is), length(Js))
    mul!(K, Xi, Xj', -2.0, zero(T))  # K = -2 X̃ X̃ᵗ
    K2 = K .+ nrm_i .+ nrm_j'
    K2 = A.ϕ.(K2)

    return K2
end

# y := A * v  (matrix-free, tiled; uses GEMM tiles under the hood)
function LinearAlgebra.mul!(
        y::AbstractVector{T}, A::StationaryKernelMatrix{T}, v::AbstractVector{T};
        iblock::Int = 1024, jblock::Int = 8192
    ) where {T <: AbstractFloat}
    n = size(A, 1)
    @assert length(y) == n == length(v)
    fill!(y, zero(T))

    # Choose backend-appropriate temporaries (Array or CuArray) by using 'similar'
    # Tile over rows I and columns J to control workspace.
    for i1 in 1:iblock:n
        i2 = min(i1 + iblock - 1, n)
        I = i1:i2
        Xi = @view A.X̃[I, :]                       # (|I|, d)
        ni = @view A.nrm[I]                         # (|I|,)

        # Workspace per-I tile (allocated on same device as Xi):
        # We’ll compute G = Xi * X̃[J,:]'  => (|I|, |J|)
        for j1 in 1:jblock:n
            j2 = min(j1 + jblock - 1, n)
            J = j1:j2
            Xj = @view A.X̃[J, :]                   # (|J|, d)
            nj = @view A.nrm[J]                     # (|J|,)
            vJ = @view v[J]

            # G = Xi * Xj' (GEMM; cuBLAS on GPU)
            G = similar(Xi, T, length(I), length(J))
            mul!(G, Xi, Xj', one(T), zero(T))

            # D = (ni .* 1ᵗ) .+ (1 .* njᵗ) .- 2G
            # Then W = ϕ.(D), and accumulate y[I] += W * v[J]
            # We fuse elementwise and gemv: compute tmp = W * vJ
            tmp = similar(ni, length(I))
            fill!(tmp, zero(T))
            @inbounds begin
                # Compute per-row dot with transformed column block to avoid forming W fully
                # Iterate columns j and accumulate tmp[i] += ϕ(ni[i]+nj[j]-2G[i,j]) * vJ[j]
                for (jj, j) in enumerate(J)
                    njj = nj[jj]
                    vjj = vJ[jj]
                    @views for ii in 1:length(I)
                        # Compute r2 and apply ϕ
                        r2 = ni[ii] + njj - 2 * G[ii, jj]
                        r2 = ifelse(r2 < zero(T), zero(T), r2)
                        tmp[ii] += A.ϕ(r2) * vjj
                    end
                end
            end
            @. tmp = tmp        # (materialize on GPU if needed)
            @views y[I] .+= tmp
        end
    end

    # Add jitter* v component (diagonal term)
    if A.jitter != zero(T)
        @inbounds @simd for i in 1:n
            y[i] += A.jitter * v[i]
        end
    end
    return y
end

# Allocate y and call mul!
function Base.:*(A::StationaryKernelMatrix{T}, v::AbstractVector{T}) where {T}
    y = similar(v, T, size(A, 1))
    return mul!(y, A, v)
end

# Materialize full K in one GEMM pass (fastest when memory permits).
# K = ϕ.( ‖X̃‖² 1ᵗ + 1 ‖X̃‖²ᵗ - 2 X̃ X̃ᵗ )  (+ jitter on diag)
function kernelmatrix(A::StationaryKernelMatrix{T}) where {T <: AbstractFloat}
    n = size(A, 1)
    X̃ = A.X̃
    nrm = A.nrm
    K = similar(X̃, T, n, n)
    mul!(K, X̃, X̃', -2.0, zero(T))  # K = -2 X̃ X̃ᵗ
    #@show K[1, 2] + nrm[1] + nrm[2]
    K2 = K .+ nrm .+ nrm'
    K2 = A.ϕ.(K2)
    #@time K2 = map(r -> (r < zero(T) ? zero(T) : A.ϕ(r) ), K2)
    #@time K = map(idx -> ((i, j) = (idx[1], idx[2]); r2 =K[i,j] + nrm[i] + nrm[j]; r2 = ifelse(r2 < zero(T), zero(T), r2);  A.ϕ(r2)), CartesianIndices((axes(K, 1), axes(K, 2))))
    #@inbounds for i in 1:n, j in 1:n
    #r2 = K[i,j] + nrm[i] + nrm[j]
    #r2 = ifelse(r2 < zero(T), zero(T), r2)
    #K[i,j] = A.ϕ(r2)
    #end
    #@inbounds for i in 1:n
    #K[i,i] += A.jitter
    #end
    return K2
end

# ——————————————————————————————————————————————————————————
# Example kernels (stationary ϕ(r2))
# Provide ready-made constructors: RBF, Matern-ν=1/2, 3/2, 5/2
# ——————————————————————————————————————————————————————————

rbfϕ(σ2::T = one(T)) where {T <: AbstractFloat} = (r2::T) -> (σ2 * exp(-T(0.5) * r2))
matern12ϕ(σ2::T = one(T)) where {T <: AbstractFloat} = (r2::T) -> ((r2 < zero(T)) ? one(T) : σ2 * exp(-sqrt(r2 + eps(T))))
function matern32ϕ(σ2::T = one(T)) where {T <: AbstractFloat}
    return (r2::T) -> begin
        r = sqrt(r2 + eps(T))
        σ2 * (1 + T(√3) * r) * exp(-T(√3) * r)
    end
end
function matern52ϕ(σ2::T = one(T)) where {T <: AbstractFloat}
    return (r2::T) -> begin
        r = sqrt(r2 + eps(T))
        a = T(√5) * r
        σ2 * (1 + a + a * a / T(3)) * exp(-a)
    end
end

# Convenience constructors with ARD lengthscales ℓ (vector or scalar).
# We scale features with s = 1 ./ ℓ so that r2 is Mahalanobis under ℓ.
function rbf_matrix(X::AbstractMatrix{T}; ℓ::Union{T, AbstractVector{T}} = one(T), σ2::T = one(T), jitter::T = T(1.0e-6)) where {T <: AbstractFloat}
    s = isa(ℓ, AbstractVector) ? (one(T) ./ ℓ) : fill(inv(ℓ), size(X, 2))
    return StationaryKernelMatrix(X, rbfϕ(σ2); s, jitter)
end

function matern32_matrix(X::AbstractMatrix{T}; ℓ::Union{T, AbstractVector{T}} = one(T), σ2::T = one(T), jitter::T = T(1.0e-6)) where {T <: AbstractFloat}
    s = isa(ℓ, AbstractVector) ? (one(T) ./ ℓ) : fill(inv(ℓ), size(X, 2))
    return StationaryKernelMatrix(X, matern32ϕ(σ2); s, jitter)
end

function matern52_matrix(X::AbstractMatrix{T}; ℓ::Union{T, AbstractVector{T}} = one(T), σ2::T = one(T), jitter::T = T(1.0e-6)) where {T <: AbstractFloat}
    s = isa(ℓ, AbstractVector) ? (one(T) ./ ℓ) : fill(inv(ℓ), size(X, 2))
    return StationaryKernelMatrix(X, matern52ϕ(σ2); s, jitter)
end

end # module
