export Domain, volume

"""
    Domain

Abstract supertype for all spatial domains in FunctionalGPs.

Subtypes must implement:
- [`volume(::Domain)`](@ref): compute the domain's volume
- `Base.in(x, ::Domain)`: test if a point is inside the domain

Concrete implementations: [`Interval`](@ref), [`BoxDomain`](@ref).
"""
abstract type Domain end

"""
    volume(d::Domain)

Compute the volume (or length/area in 1D/2D) of domain `d`.

# Examples
```julia
julia> volume(Interval(0.0, 2.0))
2.0

julia> volume(BoxDomain((0.0, 1.0), (0.0, 2.0)))
2.0
```
"""
volume(d::Domain) = throw(MethodError(volume, (d,)))
Base.in(_, d::Domain) = throw(MethodError(in, (d,)))

# Concrete domain types
include("interval.jl")
include("box.jl")
include("factorized_box.jl")
include("grids.jl")
