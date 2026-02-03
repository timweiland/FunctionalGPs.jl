export BoxDomain, volume, uniform_grid_n, uniform_grid_step

"""
    BoxDomain{T<:Real} <: Domain

An N-dimensional hyperrectangular (box) domain defined by per-dimension bounds.

# Constructors
```julia
BoxDomain((lower₁, upper₁), (lower₂, upper₂), ...)  # from tuples
BoxDomain(Interval(a, b), Interval(c, d), ...)      # from Intervals
```

# Examples
```julia
julia> box = BoxDomain((0.0, 1.0), (0.0, 2.0))
BoxDomain{Float64}(((0.0, 1.0), (0.0, 2.0)))

julia> [0.5, 1.0] in box
true

julia> volume(box)
2.0

julia> ndims(box)
2
```

See also: [`Interval`](@ref), [`uniform_grid_n`](@ref), [`FactorizedGrid`](@ref).
"""
struct BoxDomain{T <: Real} <: Domain
    bounds::Tuple{Tuple{T, T}, Vararg{Tuple{T, T}}}

    function BoxDomain(bounds::Tuple{Tuple{T, T}, Vararg{Tuple{T, T}}}) where {T <: Real}
        if !all(lower <= upper for (lower, upper) in bounds)
            throw(ArgumentError("Lower bounds may not be larger than upper bounds"))
        end
        return new{T}(bounds)
    end
end

BoxDomain() = error("Zero-dimensional box domains are unsupported")
BoxDomain(bounds::Tuple{<:Real, <:Real}...) = BoxDomain(bounds)

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

"""
    uniform_grid_n(box::BoxDomain, N₁::Int, N₂::Int, ...)

Create a [`FactorizedGrid`](@ref) over `box` with `Nᵢ` points along dimension `i`.

# Examples
```julia
julia> box = BoxDomain((0.0, 1.0), (0.0, 2.0))
julia> grid = uniform_grid_n(box, 3, 5)
FactorizedGrid((3, 5))
```

See also: [`uniform_grid_step`](@ref), [`FactorizedGrid`](@ref).
"""
function uniform_grid_n(box::BoxDomain, Ns::Int...)
    ranges = [
        range(lower; stop = upper, length = N) for
            (N, (lower, upper)) in zip(Ns, box.bounds)
    ]
    return FactorizedGrid(ranges...)
end

"""
    uniform_grid_step(box::BoxDomain, step₁::Real, step₂::Real, ...)

Create a [`FactorizedGrid`](@ref) over `box` with spacing `stepᵢ` along dimension `i`.

# Examples
```julia
julia> box = BoxDomain((0.0, 1.0), (0.0, 1.0))
julia> grid = uniform_grid_step(box, 0.5, 0.25)
FactorizedGrid((3, 5))
```

See also: [`uniform_grid_n`](@ref), [`FactorizedGrid`](@ref).
"""
function uniform_grid_step(box::BoxDomain, steps::Real...)
    ranges = [
        range(lower; stop = upper, step = step) for
            (step, (lower, upper)) in zip(steps, box.bounds)
    ]
    return FactorizedGrid(ranges...)
end
