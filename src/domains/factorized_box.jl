import KernelFunctions: ⊗

export FactorizedBoxDomains, get_intervals

"""
    FactorizedBoxDomains{N,T} <: AbstractArray{BoxDomain{T},N}

A factorized collection of N-dimensional [`BoxDomain`](@ref)s.

Stores intervals per dimension and constructs box domains on-the-fly via indexing.
Useful for representing tensor product domain decompositions without materializing
all boxes explicitly.

# Construction
```julia
FactorizedBoxDomains(intervals_dim1, intervals_dim2, ...)  # from interval vectors
intervals1 ⊗ intervals2                                    # via tensor product operator
```

# Examples
```julia
julia> x_intervals = intervals_from_endpoints([0.0, 0.5, 1.0])
julia> y_intervals = intervals_from_endpoints([0.0, 1.0])
julia> domains = x_intervals ⊗ y_intervals
2×1 FactorizedBoxDomains{2, Float64}

julia> domains[1, 1]  # access individual BoxDomain
BoxDomain{Float64}(((0.0, 0.5), (0.0, 1.0)))
```

See also: [`BoxDomain`](@ref), [`Interval`](@ref), [`get_intervals`](@ref).
"""
struct FactorizedBoxDomains{N, T} <: AbstractArray{BoxDomain{T}, N}
    interval_vecs::AbstractVector{<:AbstractVector{Interval{T}}}

    function FactorizedBoxDomains(
            interval_vecs::AbstractVector{<:AbstractVector{Interval{T}}},
        ) where {T}
        return new{length(interval_vecs), T}(interval_vecs)
    end
end

function FactorizedBoxDomains(interval_vecs::AbstractVector{Interval}...)
    return FactorizedBoxDomains([interval_vecs...])
end

Base.ndims(domains::FactorizedBoxDomains) = length(domains.interval_vecs)

Base.length(domains::FactorizedBoxDomains) = mapreduce(length, *, domains.interval_vecs)

Base.size(domains::FactorizedBoxDomains) = tuple(map(length, domains.interval_vecs)...)

function Base.getindex(domains::FactorizedBoxDomains, inds::CartesianIndex)
    d = length(domains.interval_vecs)
    if length(inds) != d
        throw(ArgumentError("Number of indices must match number of intervals"))
    end
    intervals = map(i -> domains.interval_vecs[i][inds[i]], 1:d)
    return BoxDomain(intervals...)
end

Base.getindex(domains::FactorizedBoxDomains, inds::Int...) = domains[CartesianIndex(inds...)]

function Base.getindex(domains::FactorizedBoxDomains, ind::Int)
    return domains[CartesianIndices(domains)[ind]]
end

"""
    get_intervals(domains::FactorizedBoxDomains, i::Int)

Return the vector of [`Interval`](@ref)s for dimension `i`.

    get_intervals(domains::FactorizedBoxDomains)

Return all interval vectors as a vector of vectors.
"""
get_intervals(domains::FactorizedBoxDomains, i::Int) = domains.interval_vecs[i]

get_intervals(domains::FactorizedBoxDomains) = domains.interval_vecs

"""
    intervals1 ⊗ intervals2
    domains::FactorizedBoxDomains ⊗ intervals

Construct a [`FactorizedBoxDomains`](@ref) via tensor product of interval vectors.

# Examples
```julia
julia> x = intervals_from_endpoints([0.0, 0.5, 1.0])
julia> y = intervals_from_endpoints([0.0, 1.0])
julia> domains = x ⊗ y
2×1 FactorizedBoxDomains{2, Float64}
```
"""
function ⊗(
        intervals1::AbstractVector{Interval{T}},
        intervals2::AbstractVector{Interval{T}},
    ) where {T}
    return FactorizedBoxDomains([intervals1, intervals2])
end

function ⊗(
        box_domains::FactorizedBoxDomains,
        intervals::AbstractVector{Interval{T}}
    ) where {T}
    return FactorizedBoxDomains([get_intervals(box_domains)..., intervals])
end
