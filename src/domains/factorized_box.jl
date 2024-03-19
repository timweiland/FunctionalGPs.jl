import KernelFunctions: ⊗

export FactorizedBoxDomains, get_intervals

struct FactorizedBoxDomains{N,T} <: AbstractArray{BoxDomain{T},N}
    interval_vecs::AbstractVector{<:AbstractVector{Interval{T}}}

    function FactorizedBoxDomains(
        interval_vecs::AbstractVector{<:AbstractVector{Interval{T}}},
    ) where {T}
        return new{length(interval_vecs),T}(interval_vecs)
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

get_intervals(domains::FactorizedBoxDomains, i::Int) = domains.interval_vecs[i]

get_intervals(domains::FactorizedBoxDomains) = domains.interval_vecs

function ⊗(
    intervals1::AbstractVector{Interval{T}},
    intervals2::AbstractVector{Interval{T}},
) where {T}
    return FactorizedBoxDomains([intervals1, intervals2])
end