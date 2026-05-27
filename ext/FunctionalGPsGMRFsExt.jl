module FunctionalGPsGMRFsExt

using FunctionalGPs
using FunctionalGPs:
    FunctionalGaussian, FunctionalCategory,
    INTEGRAL, FACE_INTEGRAL, EVALUATION, DERIVATIVE, OTHER,
    functional_category, get_coordinates,
    OrderingStrategy, ORDERING_PRIORITIES,
    block_range
using GaussianMarkovRandomFields
using GaussianMarkovRandomFields:
    reverse_maximin_ordering, sparsity_pattern_from_ordering,
    sparse_approximate_cholesky, sparse_approximate_cholesky!,
    form_supernodes, PermutedMatrix, GMRFMetadata, MetaGMRF, GMRF
using LinearAlgebra
using SparseArrays

export NamedBlockMetadata

"""
    NamedBlockMetadata{NT}

`GMRFMetadata` carrying the named-block layout of the `FunctionalGaussian`
that produced this GMRF. `ranges` is a `NamedTuple{block_name => UnitRange{Int}}`
giving the row/col range of each named block in the GMRF state vector.
"""
struct NamedBlockMetadata{NT <: NamedTuple} <: GMRFMetadata
    ranges::NT
end

Base.keys(m::NamedBlockMetadata) = keys(m.ranges)
function Base.show(io::IO, m::NamedBlockMetadata)
    pairs = ["$n=>$(m.ranges[n])" for n in keys(m.ranges)]
    return print(io, "NamedBlockMetadata(", join(pairs, ", "), ")")
end

# Promote a NamedBlockMetadata access to MetaGMRFs that wear it.
"""
    block_range(g::MetaGMRF{<:NamedBlockMetadata}, name::Symbol) -> UnitRange{Int}

Return the row/col range of `name` in the underlying GMRF state vector.
"""
function FunctionalGPs.block_range(
        g::MetaGMRF{<:NamedBlockMetadata}, name::Symbol,
    )
    md = g.metadata.ranges
    haskey(md, name) ||
        throw(ArgumentError("MetaGMRF has no block :$(name); has $(keys(md))"))
    return md[name]
end

"""
    nameview(g::MetaGMRF{<:NamedBlockMetadata}, x::AbstractVector) -> NamedTuple

Return a NamedTuple of views into `x`, one per named block of `g`. Lets a
DPPL `@model` body access named blocks without manually carrying
`block_range(...)` lookups:

```julia
g = vecchia(fg)
x ~ g
blocks = nameview(g, x)
blocks.u[i]      # view into x for the :u block
blocks.uxx[i]    # ditto for :uxx
```

`x` must have the same length as the GMRF.
"""
function FunctionalGPs.nameview(
        g::MetaGMRF{<:NamedBlockMetadata}, x::AbstractVector,
    )
    length(x) == length(g) || throw(
        DimensionMismatch(
            "x has length $(length(x)) but GMRF has length $(length(g))"
        )
    )
    ranges = g.metadata.ranges
    return NamedTuple{keys(ranges)}(
        ntuple(i -> view(x, ranges[keys(ranges)[i]]), length(keys(ranges)))
    )
end

# Reorder block indices by category priority. Returns a permutation of
# 1:n_blocks giving the visit order (leftmost = finest, rightmost = coarsest).
function _block_visit_order(
        categories::Vector{FunctionalCategory}, strategy::OrderingStrategy,
    )
    n = length(categories)
    if strategy === :natural
        return collect(1:n)
    end
    priorities = get(ORDERING_PRIORITIES, strategy, ORDERING_PRIORITIES[:integrals_coarsest])
    return sortperm([priorities[c] for c in categories])
end

# Glue per-block maximin orderings into a global P / ℓ following the visit
# order above. ℓ is set to the *coarsest block's* coarsest lengthscale on the
# non-coarsest blocks (matches GPFiniteVolume.jl's heuristic).
function _stitch_global_ordering(
        block_order::Vector{Int},
        block_indices::Vector{UnitRange{Int}},
        P_blocks::Vector{Vector{Int}},
        ℓ_blocks::Vector{Vector{Float64}},
    )
    n_total = isempty(block_indices) ? 0 : last(block_indices[end])
    coarsest = block_order[end]
    ℓ_coarse = ℓ_blocks[coarsest][P_blocks[coarsest][1]]

    P_global = Int[]
    ℓ_global = zeros(n_total)

    for bi in block_order
        global_idx = collect(block_indices[bi])
        P_local = P_blocks[bi]
        ℓ_local = ℓ_blocks[bi]
        if bi == coarsest
            for (li, gi) in enumerate(global_idx)
                ℓ_global[gi] = ℓ_local[li]
            end
        else
            for gi in global_idx
                ℓ_global[gi] = ℓ_coarse
            end
        end
        append!(P_global, global_idx[P_local])
    end
    return P_global, ℓ_global
end

"""
    vecchia(fg::FunctionalGaussian; ρ = 2.0, λ = 1.5, ordering = :integrals_coarsest)

Build a `MetaGMRF{NamedBlockMetadata}` approximating `fg` via KL-optimal
sparse Cholesky.

The named-block layout of `fg` is preserved: the resulting object exposes
`block_range(gmrf, :name)` for each block, in the same convention as the
`FunctionalGaussian`.

# Keyword arguments
- `ρ`: sparsity radius (larger ⇒ denser, more accurate).
- `λ`: supernodal clustering threshold; pass `nothing` for the simplicial
  factorisation.
- `ordering`: one of `:integrals_coarsest` (default), `:evaluations_coarsest`,
  `:derivatives_coarsest`, or `:natural`.
"""
function FunctionalGPs.vecchia(
        fg::FunctionalGaussian;
        ρ::Real = 2.0,
        λ::Union{Real, Nothing} = 1.5,
        ordering::OrderingStrategy = :integrals_coarsest,
    )
    names = collect(keys(fg))
    funcs = [getfield(fg, :linfuncs)[n] for n in names]
    n_blocks = length(names)

    categories = FunctionalCategory[functional_category(L) for L in funcs]
    X_blocks = Matrix{Float64}[get_coordinates(L) for L in funcs]
    block_indices = UnitRange{Int}[block_range(fg, n) for n in names]

    # Per-block maximin orderings.
    P_blocks = Vector{Vector{Int}}(undef, n_blocks)
    ℓ_blocks = Vector{Vector{Float64}}(undef, n_blocks)
    for i in 1:n_blocks
        P_blocks[i], ℓ_blocks[i] = reverse_maximin_ordering(X_blocks[i])
    end

    block_order = _block_visit_order(categories, ordering)
    P_global, ℓ_global = _stitch_global_ordering(
        block_order, block_indices, P_blocks, ℓ_blocks,
    )

    # Global coordinates in the FunctionalGaussian's natural order.
    X_global = hcat(X_blocks...)

    # Joint covariance from the FunctionalGaussian; symmetrise.
    K = Symmetric(Matrix(cov(fg)))

    S = sparsity_pattern_from_ordering(X_global, P_global, ℓ_global, Float64(ρ))
    K_P = PermutedMatrix(K, P_global)
    if λ === nothing
        sparse_approximate_cholesky!(K_P, S)
        L = S
    else
        sc = form_supernodes(S, P_global, ℓ_global; λ = Float64(λ))
        L = sparse_approximate_cholesky(K_P, sc)
    end

    Q_perm = L * L'
    P_inv = invperm(P_global)
    Q = Q_perm[P_inv, P_inv]

    n_total = sum(length, block_indices)
    # Use the kernel-induced eltype so the mean propagates ForwardDiff Duals
    # (or any other AD eltype) when the kernel is parameterised by a hyperparam.
    inner = GMRF(zeros(eltype(Q), n_total), Q)
    md = NamedBlockMetadata(NamedTuple{Tuple(names)}(Tuple(block_indices)))
    return MetaGMRF(inner, md)
end

end # module
