export Box, volume, uniform_grid_n, uniform_grid_step

struct Box{T<:Real} <: Domain
    bounds::Tuple{Tuple{T, T}, Vararg{Tuple{T, T}}}

    function Box(bounds::Tuple{Tuple{T, T}, Vararg{Tuple{T, T}}}) where {T<:Real}
        if !all(lower <= upper for (lower, upper) in bounds)
            throw(ArgumentError("Lower bounds may not be larger than upper bounds"))
        end
        return new{T}(bounds)
    end
end

function Box(bounds::Tuple{<:Real, <:Real}...)
    return Box(bounds)
end

Base.ndims(box::Box) = length(box.bounds)
Base.getindex(box::Box, i::Integer) = box.bounds[i]
Base.getindex(box::Box, i::Integer, j::Integer) = box.bounds[i][j]

volume(box::Box) = prod(upper - lower for (lower, upper) in box.bounds)
function Base.in(x::AbstractVector, box::Box)
    return all(lower <= x[i] <= upper for (i, (lower, upper)) in enumerate(box.bounds))
end

function uniform_grid_n(box::Box, Ns::Int...)
    ranges = [range(lower, stop=upper, length=N) for (N, (lower, upper)) in zip(Ns, box.bounds)]
    return FactorizedGrid(ranges...)
end

function uniform_grid_step(box::Box, steps::Real...)
    ranges = [range(lower, stop=upper, step=step) for (step, (lower, upper)) in zip(steps, box.bounds)]
    return FactorizedGrid(ranges...)
end
