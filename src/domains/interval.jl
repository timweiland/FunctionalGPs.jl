export Interval, volume, uniform_grid_n, uniform_grid_step, intervals_from_endpoints

struct Interval{T<:Real} <: Domain
    lower::T
    upper::T

    function Interval(lower::T, upper::T) where {T<:Real}
        if !(lower <= upper)
            throw(ArgumentError("Lower bound may not be larger than upper bound"))
        end
        return new{T}(lower, upper)
    end
end

Base.ndims(interval::Interval) = 1
function Base.getindex(interval::Interval, i::Integer)
    if i ∉ (1, 2)
        throw(ArgumentError("Index out of bounds"))
    end
    return i == 1 ? interval.lower : interval.upper
end

volume(interval::Interval) = interval.upper - interval.lower
Base.in(x::Number, interval::Interval) = interval.lower <= x <= interval.upper

Base.isequal(a::Interval, b::Interval) = a.lower == b.lower && a.upper == b.upper

function uniform_grid_n(interval::Interval, N::Int)
    return range(interval.lower; stop = interval.upper, length = N)
end

function uniform_grid_step(interval::Interval, step::Real)
    return range(interval.lower; stop = interval.upper, step = step)
end

function intervals_from_endpoints(endpoints::AbstractVector)
    return [Interval(endpoints[i], endpoints[i+1]) for i in 1:length(endpoints)-1]
end
