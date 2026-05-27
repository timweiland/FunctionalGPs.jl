export VectorizedLebesgueIntegral, IntegralFunctional

"""
    VectorizedLebesgueIntegral{T <: Domain} <: AbstractLinearFunctional

Integration functional that computes integrals over one or more domains.

This functional represents the Lebesgue integral ∫_D f(x) dx for each domain D
in the provided collection. It is useful for Bayesian quadrature, cell-averaged
observations, and integral constraints.

# Fields
- `domains::AbstractArray{T}`: Array of domains to integrate over

# Example
```julia
using FunctionalGPs

# Single interval integration
∫ = VectorizedLebesgueIntegral(Interval(0.0, 1.0))

# Multiple intervals (vectorized)
intervals = [Interval(0.0, 1.0), Interval(1.0, 2.0), Interval(2.0, 3.0)]
∫_vec = VectorizedLebesgueIntegral(intervals)

# 2D box integration
box = BoxDomain((0.0, 1.0), (0.0, 1.0))
∫_2d = VectorizedLebesgueIntegral(box)
```

# See also
- [`EvaluationFunctional`](@ref): For point evaluation
- [`TensorProductFunctional`](@ref): For combining with evaluation on other dimensions
"""
struct VectorizedLebesgueIntegral{T <: Domain} <: AbstractLinearFunctional
    domains::AbstractArray{T}

    function VectorizedLebesgueIntegral(domains::AbstractArray{T}) where {T}
        if Base.length(domains) == 0
            throw(ArgumentError("At least one domain must be provided"))
        end
        return new{T}(domains)
    end

    """
        VectorizedLebesgueIntegral(domains...)

    Construct an integration functional from individual domains.
    """
    VectorizedLebesgueIntegral(domains::Domain...) =
        VectorizedLebesgueIntegral(collect(domains))
end

output_shape(ℒ::VectorizedLebesgueIntegral) = size(ℒ.domains)

"""
    IntegralFunctional

Alias for [`VectorizedLebesgueIntegral`](@ref). The shorter name reads
better alongside `EvaluationFunctional` and `PartialDerivative` in
narrative code.
"""
const IntegralFunctional = VectorizedLebesgueIntegral
