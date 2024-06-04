# TODO implement the Sphere domain


# Note: We assume the center to be at (0,0,0)
struct SphereDomain{T<:Real} <: Domain
    radius::T

    function SphereDomain(radius::T) where {T<:Real}
        if radius < 0
            throw(ArgumentError("Radius must be non-negative"))
        end
        return new{T}(radius)
    end
end

SphereDomain() = error("Radius must be specified")

function SphereDomain(radius::Real)
    return SphereDomain(radius)
end

Base.ndims(sphere::SphereDomain) = 3

function Base.getindex(sphere::SphereDomain, i::Integer)
    if i != 1
        throw(ArgumentError("Index out of bounds"))
    end
    return sphere.radius
end


volume(sphere::SphereDomain) = 4/3 * π * sphere.radius^3

# Checks whether a point is ON the sphere
# here we assume a point in R^3 as input
function Base.in(x::AbstractVector, sphere::SphereDomain)
    return norm(x) == sphere.radius
end

Base.isequal(a::SphereDomain, b::SphereDomain) = a.radius == b.radius

# TODO grid
