export Interval, volume, uniform_grid_n, uniform_grid_step, intervals_from_endpoints

"""
    Interval{T<:Real} <: Domain

A one-dimensional closed interval `[lower, upper]`.

# Fields
- `lower::T`: the left endpoint
- `upper::T`: the right endpoint

# Examples
```julia
julia> I = Interval(0.0, 1.0)
Interval{Float64}(0.0, 1.0)

julia> 0.5 in I
true

julia> volume(I)
1.0
```

See also: [`BoxDomain`](@ref), [`uniform_grid_n`](@ref), [`uniform_grid_step`](@ref).
"""
struct Interval{T <: Real} <: Domain
    lower::T
    upper::T

    function Interval(lower::T, upper::T) where {T <: Real}
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

"""
    uniform_grid_n(interval::Interval, N::Int)

Create a uniform grid of `N` points over `interval`.

Returns a `StepRangeLen` spanning from `interval.lower` to `interval.upper`.

# Examples
```julia
julia> uniform_grid_n(Interval(0.0, 1.0), 5)
0.0:0.25:1.0
```

See also: [`uniform_grid_step`](@ref).
"""
function uniform_grid_n(interval::Interval, N::Int)
    return range(interval.lower; stop = interval.upper, length = N)
end

"""
    uniform_grid_step(interval::Interval, step::Real)

Create a uniform grid over `interval` with spacing `step`.

Returns a `StepRangeLen` starting at `interval.lower` with the given step size.

# Examples
```julia
julia> uniform_grid_step(Interval(0.0, 1.0), 0.2)
0.0:0.2:1.0
```

See also: [`uniform_grid_n`](@ref).
"""
function uniform_grid_step(interval::Interval, step::Real)
    return range(interval.lower; stop = interval.upper, step = step)
end

"""
    intervals_from_endpoints(endpoints::AbstractVector)

Construct a vector of adjacent [`Interval`](@ref)s from a sorted vector of endpoints.

Given `[a, b, c, ...]`, returns `[Interval(a,b), Interval(b,c), ...]`.

# Examples
```julia
julia> intervals_from_endpoints([0.0, 0.5, 1.0])
2-element Vector{Interval{Float64}}:
 Interval{Float64}(0.0, 0.5)
 Interval{Float64}(0.5, 1.0)
```
"""
function intervals_from_endpoints(endpoints::AbstractVector)
    return [Interval(endpoints[i], endpoints[i + 1]) for i in 1:(Base.length(endpoints) - 1)]
end
