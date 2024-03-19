export BoxDomain, volume, uniform_grid_n, uniform_grid_step

struct BoxDomain{T<:Real} <: Domain
    bounds::Tuple{Tuple{T,T},Vararg{Tuple{T,T}}}

    function BoxDomain(bounds::Tuple{Tuple{T,T},Vararg{Tuple{T,T}}}) where {T<:Real}
        if !all(lower <= upper for (lower, upper) in bounds)
            throw(ArgumentError("Lower bounds may not be larger than upper bounds"))
        end
        return new{T}(bounds)
    end
end

BoxDomain() = error("Zero-dimensional box domains are unsupported")
BoxDomain(bounds::Tuple{<:Real,<:Real}...) = BoxDomain(bounds)

function BoxDomain(intervals::Interval{<:Real}...)
    return BoxDomain(map(interval -> (interval.lower, interval.upper), intervals))
end

Base.ndims(box::BoxDomain) = length(box.bounds)
Base.getindex(box::BoxDomain, i::Integer) = box.bounds[i]
Base.getindex(box::BoxDomain, i::Integer, j::Integer) = box.bounds[i][j]

volume(box::BoxDomain) = prod(upper - lower for (lower, upper) in box.bounds)
function Base.in(x::AbstractVector, box::BoxDomain)
    return all(lower <= x[i] <= upper for (i, (lower, upper)) in enumerate(box.bounds))
end

function uniform_grid_n(box::BoxDomain, Ns::Int...)
    ranges = [
        range(lower; stop = upper, length = N) for
        (N, (lower, upper)) in zip(Ns, box.bounds)
    ]
    return FactorizedGrid(ranges...)
end

function uniform_grid_step(box::BoxDomain, steps::Real...)
    ranges = [
        range(lower; stop = upper, step = step) for
        (step, (lower, upper)) in zip(steps, box.bounds)
    ]
    return FactorizedGrid(ranges...)
end
