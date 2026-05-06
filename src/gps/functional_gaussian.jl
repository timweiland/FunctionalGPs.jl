import AbstractGPs: AbstractGP
import Distributions: AbstractMvNormal, MvNormal, logpdf, loglikelihood
import Distributions
import Statistics: mean, cov, var
using LinearAlgebra
using BlockArrays
import Random: AbstractRNG

export FunctionalGaussian, LazyMvNormal, block_range, marginal, as_mvn, condition

"""
    LazyMvNormal{T, M, S} <: AbstractMvNormal

Multivariate normal distribution that does not eagerly factorise its
covariance. Stores the mean and the covariance matrix as-is, so structured
covariance types (Toeplitz, Khatri-Rao, sparse, etc.) keep their fast
`getindex`/matvec paths intact.

Returned by `marginal(fg, name)` and the `fg.<name>` property accessor on a
[`FunctionalGaussian`](@ref).

# Cheap, structure-preserving
- `mean(d)`, `cov(d)`, `var(d)`, `length(d)` — no factorisation.
- `cov(d)` returns the underlying matrix unchanged (e.g. `StationaryKernelMatrix`).

# Heavier (materialises and factorises on each call)
- `logpdf(d, x)`, `rand(d)`, `Distributions.logdetcov(d)`, `sqmahal(d, x)` —
  fall back to `MvNormal(d)`. For repeated likelihood evaluation, build the
  factorised form once via `MvNormal(d)` and reuse it.

# Conversion
- `MvNormal(d)` materialises the covariance and factorises eagerly, returning a
  regular `Distributions.MvNormal`. Use this when you want PDMat-style caching.
"""
struct LazyMvNormal{T <: Real, M <: AbstractVector{T}, S <: AbstractMatrix{T}} <:
    AbstractMvNormal
    μ::M
    Σ::S

    function LazyMvNormal(μ::AbstractVector{T}, Σ::AbstractMatrix{T}) where {T <: Real}
        size(Σ, 1) == size(Σ, 2) ||
            throw(DimensionMismatch("covariance must be square, got $(size(Σ))"))
        length(μ) == size(Σ, 1) || throw(
            DimensionMismatch(
                "mean length $(length(μ)) does not match covariance size $(size(Σ))"
            )
        )
        return new{T, typeof(μ), typeof(Σ)}(μ, Σ)
    end
end

function LazyMvNormal(μ::AbstractVector, Σ::AbstractMatrix)
    T = promote_type(eltype(μ), eltype(Σ))
    return LazyMvNormal(convert(AbstractVector{T}, μ), convert(AbstractMatrix{T}, Σ))
end

Base.length(d::LazyMvNormal) = length(d.μ)
Base.eltype(::Type{<:LazyMvNormal{T}}) where {T} = T
Distributions.dim(d::LazyMvNormal) = length(d.μ)
Distributions.params(d::LazyMvNormal) = (d.μ, d.Σ)

mean(d::LazyMvNormal) = d.μ
cov(d::LazyMvNormal) = d.Σ
var(d::LazyMvNormal) = diag(d.Σ)

"""
    MvNormal(d::LazyMvNormal) -> MvNormal

Materialise a `LazyMvNormal` into a regular `Distributions.MvNormal`,
factorising the covariance eagerly. Use this when you need to call `logpdf` /
`rand` repeatedly and want to amortise the factorisation cost.
"""
function MvNormal(d::LazyMvNormal)
    Σ = Matrix(d.Σ)
    Σ = (Σ + Σ') / 2
    return MvNormal(Vector(d.μ), Σ)
end

# Heavier ops fall back to materialising. These do NOT cache — for repeated
# evaluation, the user should call `MvNormal(d)` once and reuse the result.
Distributions._logpdf(d::LazyMvNormal, x::AbstractVector) =
    Distributions._logpdf(MvNormal(d), x)
Distributions.logdetcov(d::LazyMvNormal) = Distributions.logdetcov(MvNormal(d))
Distributions.sqmahal(d::LazyMvNormal, x::AbstractVector) =
    Distributions.sqmahal(MvNormal(d), x)
Distributions.invcov(d::LazyMvNormal) = Distributions.invcov(MvNormal(d))
Distributions._rand!(rng::AbstractRNG, d::LazyMvNormal, x::AbstractVector) =
    Distributions._rand!(rng, MvNormal(d), x)

function Base.show(io::IO, d::LazyMvNormal)
    return print(io, "LazyMvNormal(dim=", length(d), ", cov=", nameof(typeof(d.Σ)), ")")
end

"""
    FunctionalGaussian(f::AbstractGP; name1 = ℒ1, name2 = ℒ2, ...)
    FunctionalGaussian(f::AbstractGP, linfuncs::NamedTuple)

A finite-dimensional Gaussian induced by applying a collection of named linear
functionals to a Gaussian process.

This is the package-native object that owns the joint Gaussian algebra over
GP-derived random variables. Marginalising, conditioning, and likelihood
evaluation are all operations on this object — never on the individual blocks
in isolation, because doing so would drop cross-covariances induced by the
shared underlying GP.

# Fields
- `f`: the underlying GP
- `linfuncs`: NamedTuple of name => `AbstractLinearFunctional`
- `μ`: joint mean vector (concatenated block means)
- `Σ`: joint covariance as a `BlockMatrix`
- `ranges`: NamedTuple of name => `UnitRange{Int}` giving the row/column range
  of each named block in `μ`/`Σ`

# Example
```julia
k = WendlandKernel(1, 3, 8 // 10)
f = GP(k)

L_y  = EvaluationFunctional(X)
L_dy = EvaluationFunctional(Xd) ∘ PartialDerivative((1,))
fg = FunctionalGaussian(f; y = L_y, dy = L_dy)

mean(fg)              # joint mean (length 5)
cov(fg, :y, :dy)      # cross-covariance block
marginal(fg, :dy)     # MvNormal over the dy block
post = condition(fg, (; y = y_obs); noise = (; y = σ²))
loglikelihood(fg, (; y = y_obs); noise = (; y = σ²))
```
"""
struct FunctionalGaussian{F <: AbstractGP, NT <: NamedTuple, M, S, R <: NamedTuple}
    f::F
    linfuncs::NT
    μ::M
    Σ::S
    ranges::R
end

function FunctionalGaussian(f::AbstractGP, linfuncs::NamedTuple)
    isempty(linfuncs) && throw(ArgumentError("FunctionalGaussian needs at least one functional"))
    names = keys(linfuncs)
    funcs = values(linfuncs)
    stacked = StackedLinearFunctional(funcs...)
    μ = stacked(f.mean)
    Σ = stacked(stacked(f.kernel))
    lengths = map(lf -> prod(output_shape(lf)), funcs)
    ranges = _build_ranges(names, lengths)
    return FunctionalGaussian(f, linfuncs, μ, Σ, ranges)
end

function FunctionalGaussian(f::AbstractGP; linfuncs...)
    return FunctionalGaussian(f, NamedTuple(linfuncs))
end

function _build_ranges(names::NTuple{N, Symbol}, lengths::NTuple{N, Int}) where {N}
    offset = 0
    rs = ntuple(N) do i
        r = (offset + 1):(offset + lengths[i])
        offset += lengths[i]
        return r
    end
    return NamedTuple{names}(rs)
end

Base.keys(fg::FunctionalGaussian) = keys(getfield(fg, :linfuncs))
Base.length(fg::FunctionalGaussian) = length(getfield(fg, :μ))

"""
    getproperty(fg::FunctionalGaussian, name::Symbol)

Property access on a `FunctionalGaussian`:

- If `name` is a block name (one of `keys(fg)`), returns `marginal(fg, name)` —
  the marginal [`LazyMvNormal`](@ref) over that block. The covariance is *not*
  factorised: `cov(fg.<name>)` returns the underlying structured matrix
  (`StationaryKernelMatrix`, sparse, etc.) — same object as `cov(fg, name)`.
- Otherwise falls back to ordinary field access.

Block names take priority over the underlying struct fields. To access an
internal field that is shadowed by a block name (e.g. a block named `:μ`),
use `getfield(fg, :μ)`.
"""
function Base.getproperty(fg::FunctionalGaussian, name::Symbol)
    if haskey(getfield(fg, :ranges), name)
        return marginal(fg, name)
    end
    return getfield(fg, name)
end

function Base.propertynames(fg::FunctionalGaussian, private::Bool = false)
    block_names = keys(fg)
    if private
        return (block_names..., fieldnames(typeof(fg))...)
    end
    return block_names
end

"""
    block_range(fg::FunctionalGaussian, name::Symbol) -> UnitRange{Int}

Return the index range of the named block in the joint mean/covariance.
"""
function block_range(fg::FunctionalGaussian, name::Symbol)
    haskey(getfield(fg, :ranges), name) ||
        throw(ArgumentError("FunctionalGaussian has no block :$(name); has $(keys(fg))"))
    return getfield(fg, :ranges)[name]
end

"""
    mean(fg::FunctionalGaussian) -> Vector
    mean(fg::FunctionalGaussian, name::Symbol) -> Vector

Return the joint mean, or the mean of a named block.
"""
mean(fg::FunctionalGaussian) = getfield(fg, :μ)
mean(fg::FunctionalGaussian, name::Symbol) = view(getfield(fg, :μ), block_range(fg, name))

"""
    cov(fg::FunctionalGaussian) -> AbstractMatrix
    cov(fg::FunctionalGaussian, name::Symbol) -> AbstractMatrix
    cov(fg::FunctionalGaussian, a::Symbol, b::Symbol) -> AbstractMatrix

Return the joint covariance, the self-covariance of a named block, or the
cross-covariance between two named blocks.
"""
cov(fg::FunctionalGaussian) = getfield(fg, :Σ)
cov(fg::FunctionalGaussian, name::Symbol) = cov(fg, name, name)
function cov(fg::FunctionalGaussian, a::Symbol, b::Symbol)
    return blocks(getfield(fg, :Σ))[_block_index(fg, a), _block_index(fg, b)]
end

function _block_index(fg::FunctionalGaussian, name::Symbol)
    haskey(getfield(fg, :ranges), name) ||
        throw(ArgumentError("FunctionalGaussian has no block :$(name); has $(keys(fg))"))
    return findfirst(==(name), keys(fg))::Int
end

function Base.show(io::IO, fg::FunctionalGaussian)
    print(io, "FunctionalGaussian(")
    first = true
    for name in keys(fg)
        first || print(io, ", ")
        first = false
        print(io, name, " [", length(block_range(fg, name)), "]")
    end
    return print(io, ")")
end

function Base.show(io::IO, ::MIME"text/plain", fg::FunctionalGaussian)
    println(io, "FunctionalGaussian (length ", length(fg), ", ", length(keys(fg)), " blocks)")
    println(io, "  kernel: ", nameof(typeof(getfield(fg, :f).kernel)))
    name_w = maximum(length ∘ string, keys(fg))
    rng_w = ndigits(length(fg))
    ctx = IOContext(io, :compact => true, :limit => true)
    for name in keys(fg)
        rng = block_range(fg, name)
        rng_str = lpad(string(first(rng), ":", last(rng)), 2 * rng_w + 1)
        print(io, "  :", rpad(string(name), name_w), "  ", rng_str, "  ")
        show(ctx, getfield(fg, :linfuncs)[name])
        println(io)
    end
    return nothing
end

"""
    as_mvn(fg::FunctionalGaussian) -> MvNormal

Return the joint distribution as a flat `MvNormal`.
"""
function as_mvn(fg::FunctionalGaussian)
    Σ = Matrix(getfield(fg, :Σ))
    Σ = (Σ + Σ') / 2
    return MvNormal(Vector(getfield(fg, :μ)), Σ)
end

"""
    marginal(fg::FunctionalGaussian, name::Symbol) -> LazyMvNormal
    marginal(fg::FunctionalGaussian, names::Tuple{Vararg{Symbol}}) -> LazyMvNormal

Return the marginal distribution over one or more named blocks, in the order
given. The returned [`LazyMvNormal`](@ref) does not factorise its covariance,
so structured per-block matrices keep their representation.
"""
marginal(fg::FunctionalGaussian, name::Symbol) = _single_block_marginal(fg, name)
function marginal(fg::FunctionalGaussian, names::Tuple{Vararg{Symbol}})
    _check_blocks(fg, names, "marginal")
    if length(names) == 1
        return _single_block_marginal(fg, names[1])
    end
    bs = map(n -> _block_index(fg, n), names)
    idx = _concat_ranges(fg, names)
    μ = getfield(fg, :μ)[idx]
    Σ = _assemble_dense(fg, bs, bs)
    Σ = (Σ + Σ') / 2
    return LazyMvNormal(μ, Σ)
end

# Single-block marginals keep the underlying CovarianceMatrix subtype untouched
# (no copy, no symmetrisation) — one block is symmetric by construction.
function _single_block_marginal(fg::FunctionalGaussian, name::Symbol)
    rng = block_range(fg, name)   # validates `name`
    μ = view(getfield(fg, :μ), rng)
    Σ = cov(fg, name)
    return LazyMvNormal(μ, Σ)
end

function _concat_ranges(fg::FunctionalGaussian, names::Tuple{Vararg{Symbol}})
    isempty(names) && throw(ArgumentError("need at least one block name"))
    return reduce(vcat, (collect(block_range(fg, n)) for n in names))
end

"""
    condition(fg::FunctionalGaussian, observed::NamedTuple; noise::NamedTuple = (;)) -> NamedTuple

Condition the joint Gaussian on observed values for a subset of named blocks
and return the conditional distributions over the remaining blocks.

`observed` is a NamedTuple mapping block names to observed values. `noise`
optionally maps observed block names to additive Gaussian noise (a scalar
variance, a vector of variances, or a covariance matrix). Blocks not present
in `noise` are treated as noise-free.

Returns a NamedTuple with one entry per latent (i.e., not observed) block;
each entry is a [`LazyMvNormal`](@ref). The latent posterior covariance comes
out of a Cholesky solve and is materially dense, but `LazyMvNormal` defers
factorisation so `mean`/`cov`/`var` access remains cheap.
"""
function condition(
        fg::FunctionalGaussian,
        observed::NamedTuple;
        noise::NamedTuple = NamedTuple(),
    )
    obs_names = keys(observed)
    _check_blocks(fg, obs_names, "observed")
    for n in keys(noise)
        n in obs_names ||
            throw(ArgumentError("noise specified for :$n but :$n is not observed"))
    end

    lat_names = Tuple(n for n in keys(fg) if !(n in obs_names))
    isempty(lat_names) && throw(ArgumentError("all blocks observed; nothing to condition over"))

    obs_blocks = map(n -> _block_index(fg, n), obs_names)
    lat_blocks = map(n -> _block_index(fg, n), lat_names)
    obs_idx = _concat_ranges(fg, obs_names)
    lat_idx = _concat_ranges(fg, lat_names)

    μ_o = getfield(fg, :μ)[obs_idx]
    μ_l = getfield(fg, :μ)[lat_idx]

    Σ_oo = _assemble_dense(fg, obs_blocks, obs_blocks)
    Σ_oo = Σ_oo + _stacked_noise_cov(fg, obs_names, noise)
    Σ_oo = (Σ_oo + Σ_oo') / 2

    y = _stacked_obs(observed, obs_names)
    C = cholesky(Symmetric(Σ_oo))

    # Mean update: Σ_lo * (C \ residual). Keep Σ_lo as block-structured so each
    # Bs[i,j] * subvector hits the block's native fast matvec where available
    # (Toeplitz, KhatriRao, etc.).
    residual = C \ (y - μ_o)
    μ_post = μ_l + _block_matmul(fg, lat_blocks, obs_blocks, residual)

    # Covariance update: Σ_lo * (C \ Σ_ol). Cholesky's solve needs a dense
    # right-hand side, but we apply Σ_lo block-aware on the result.
    Σ_ol = _assemble_dense(fg, obs_blocks, lat_blocks)
    M = C \ Σ_ol
    Σ_ll = _assemble_dense(fg, lat_blocks, lat_blocks)
    Σ_post = Σ_ll - _block_matmul(fg, lat_blocks, obs_blocks, M)
    Σ_post = (Σ_post + Σ_post') / 2

    return _split_posterior(fg, lat_names, μ_post, Σ_post)
end

"""
    loglikelihood(fg::FunctionalGaussian, observed::NamedTuple; noise::NamedTuple = (;)) -> Real

Marginal log-likelihood of the observed block(s) under the joint Gaussian.

Equivalent to `logpdf(marginal(fg, observed_names), stacked_observations)`,
plus optional additive noise on the observed blocks.
"""
function loglikelihood(
        fg::FunctionalGaussian,
        observed::NamedTuple;
        noise::NamedTuple = NamedTuple(),
    )
    obs_names = keys(observed)
    _check_blocks(fg, obs_names, "observed")
    for n in keys(noise)
        n in obs_names ||
            throw(ArgumentError("noise specified for :$n but :$n is not observed"))
    end

    obs_blocks = map(n -> _block_index(fg, n), obs_names)
    obs_idx = _concat_ranges(fg, obs_names)
    μ_o = getfield(fg, :μ)[obs_idx]
    Σ_oo = _assemble_dense(fg, obs_blocks, obs_blocks)
    Σ_oo = Σ_oo + _stacked_noise_cov(fg, obs_names, noise)
    Σ_oo = (Σ_oo + Σ_oo') / 2

    y = _stacked_obs(observed, obs_names)
    return logpdf(MvNormal(μ_o, Σ_oo), y)
end

function _check_blocks(fg::FunctionalGaussian, names, role::AbstractString)
    for n in names
        haskey(getfield(fg, :ranges), n) ||
            throw(ArgumentError("$role block :$n is not in $(keys(fg))"))
    end
    return nothing
end

function _stacked_obs(observed::NamedTuple, names::Tuple{Vararg{Symbol}})
    return reduce(vcat, (vec(observed[n]) for n in names))
end

function _stacked_noise_cov(
        fg::FunctionalGaussian,
        obs_names::Tuple{Vararg{Symbol}},
        noise::NamedTuple,
    )
    sizes = map(n -> length(block_range(fg, n)), obs_names)
    total = sum(sizes)
    T = _noise_eltype(getfield(fg, :Σ), noise)
    Σ = zeros(T, total, total)
    offset = 0
    for (n, sz) in zip(obs_names, sizes)
        if haskey(noise, n)
            block = _noise_block(noise[n], sz)
            Σ[(offset + 1):(offset + sz), (offset + 1):(offset + sz)] .+= block
        end
        offset += sz
    end
    return Σ
end

# Pick an element type wide enough to hold both the joint covariance and any
# user-supplied noise. Keeping this generic is what lets `loglikelihood` /
# `condition` work under ForwardDiff Duals or BigFloat hyperparameters.
_noise_eltype(Σ, noise::NamedTuple) = isempty(noise) ? eltype(Σ) :
    promote_type(eltype(Σ), map(_eltype_of, values(noise))...)
_eltype_of(x::Real) = typeof(x)
_eltype_of(x::AbstractArray) = eltype(x)

"""
    _assemble_dense(fg, row_blocks, col_blocks) -> Matrix

Materialise the dense submatrix `Σ[row_blocks, col_blocks]` from the underlying
block storage. Each block is converted to dense via `Matrix(...)` once, so any
fast bulk-conversion path defined by the block type (e.g. for stationary or
Khatri-Rao matrices) is used.
"""
function _assemble_dense(fg::FunctionalGaussian, row_blocks, col_blocks)
    Bs = blocks(getfield(fg, :Σ))
    names = keys(fg)
    row_lens = map(i -> length(block_range(fg, names[i])), row_blocks)
    col_lens = map(j -> length(block_range(fg, names[j])), col_blocks)
    nrow = sum(row_lens)
    ncol = sum(col_lens)
    M = Matrix{eltype(getfield(fg, :Σ))}(undef, nrow, ncol)
    row_off = 0
    for (ri_idx, ri) in enumerate(row_blocks)
        rl = row_lens[ri_idx]
        col_off = 0
        for (cj_idx, cj) in enumerate(col_blocks)
            cl = col_lens[cj_idx]
            block = Bs[ri, cj]
            M[(row_off + 1):(row_off + rl), (col_off + 1):(col_off + cl)] = Matrix(block)
            col_off += cl
        end
        row_off += rl
    end
    return M
end

"""
    _block_matmul(fg, row_blocks, col_blocks, X) -> Vector or Matrix

Compute `Σ[row_blocks, col_blocks] * X` block-by-block, dispatching to each
block's native `*` implementation. This preserves fast matvec/matmul paths for
structured `CovarianceMatrix` subtypes (e.g. Toeplitz, Khatri-Rao) without
materialising the cross-block submatrix.
"""
function _block_matmul(
        fg::FunctionalGaussian,
        row_blocks,
        col_blocks,
        X::AbstractVecOrMat,
    )
    Bs = blocks(getfield(fg, :Σ))
    names = keys(fg)
    row_lens = map(i -> length(block_range(fg, names[i])), row_blocks)
    col_lens = map(j -> length(block_range(fg, names[j])), col_blocks)
    nrow = isempty(row_lens) ? 0 : sum(row_lens)
    ncol = isempty(col_lens) ? 0 : sum(col_lens)
    size(X, 1) == ncol || throw(
        DimensionMismatch(
            "RHS has $(size(X, 1)) rows but column blocks total $(ncol)"
        )
    )
    if X isa AbstractVector
        out = zeros(promote_type(eltype(getfield(fg, :Σ)), eltype(X)), nrow)
    else
        out = zeros(promote_type(eltype(getfield(fg, :Σ)), eltype(X)), nrow, size(X, 2))
    end
    row_off = 0
    for (ri_idx, ri) in enumerate(row_blocks)
        rl = row_lens[ri_idx]
        out_rows = (row_off + 1):(row_off + rl)
        col_off = 0
        for (cj_idx, cj) in enumerate(col_blocks)
            cl = col_lens[cj_idx]
            in_rows = (col_off + 1):(col_off + cl)
            block = Bs[ri, cj]
            if X isa AbstractVector
                out[out_rows] .+= block * view(X, in_rows)
            else
                out[out_rows, :] .+= block * view(X, in_rows, :)
            end
            col_off += cl
        end
        row_off += rl
    end
    return out
end

_noise_block(σ²::Real, n::Int) = Matrix(σ² * I(n))
function _noise_block(v::AbstractVector, n::Int)
    length(v) == n ||
        throw(DimensionMismatch("noise vector length $(length(v)) ≠ block length $n"))
    return Matrix(Diagonal(v))
end
function _noise_block(M::AbstractMatrix, n::Int)
    size(M) == (n, n) ||
        throw(DimensionMismatch("noise matrix size $(size(M)) ≠ ($n, $n)"))
    return Matrix(M)
end

function _split_posterior(
        fg::FunctionalGaussian,
        lat_names::Tuple{Vararg{Symbol}},
        μ_post::AbstractVector,
        Σ_post::AbstractMatrix,
    )
    sizes = map(n -> length(block_range(fg, n)), lat_names)
    offsets = cumsum((0, sizes[1:(end - 1)]...))
    posts = ntuple(length(lat_names)) do i
        rng = (offsets[i] + 1):(offsets[i] + sizes[i])
        μ = μ_post[rng]
        Σ = Σ_post[rng, rng]
        Σ = (Σ + Σ') / 2
        return LazyMvNormal(μ, Σ)
    end
    return NamedTuple{lat_names}(posts)
end
