export AbstractLinFctlLinFuncOpConcat, LinFctlLinFuncOpConcat

"""
    AbstractLinFctlLinFuncOpConcat{N} <: AbstractLinearFunctional

Abstract type for a linear functional composed with function operators.

Compositions are created using the `∘` operator, e.g., `δ ∘ ∂x` for
evaluating the derivative.

# See also
- [`LinFctlLinFuncOpConcat`](@ref): Concrete implementation
"""
abstract type AbstractLinFctlLinFuncOpConcat{N} <: AbstractLinearFunctional end

"""
    linfctl(ℒ::AbstractLinFctlLinFuncOpConcat)

Return the base linear functional (the leftmost part of the composition).
"""
linfctl(op::AbstractLinFctlLinFuncOpConcat) = op.linfctl

"""
    linfuncops(ℒ::AbstractLinFctlLinFuncOpConcat)

Return the tuple of function operators applied before the functional.
"""
linfuncops(op::AbstractLinFctlLinFuncOpConcat) = op.linfuncops

output_shape(op::AbstractLinFctlLinFuncOpConcat) = output_shape(linfctl(op))

function Base.show(io::IO, op::AbstractLinFctlLinFuncOpConcat)
    return print(
        io,
        "$(string(linfctl(op))) ∘ " *
            join(["($(string(linfuncop)))" for linfuncop in reverse(linfuncops(op))], " ∘ "),
    )
end

"""
    LinFctlLinFuncOpConcat{N} <: AbstractLinFctlLinFuncOpConcat{N}

Composition of a linear functional with N function operators, created using `∘`.

This represents the operation ℒ ∘ D₁ ∘ D₂ ∘ ... ∘ Dₙ, where ℒ is a functional
and Dᵢ are operators like differentiation. Used for derivative observations
and physics-informed constraints.

# Fields
- `linfctl::AbstractLinearFunctional`: The base functional
- `linfuncops::NTuple{N, AbstractLinearFunctionOperator}`: The operators to apply

# Example
```julia
# Evaluate the first derivative at specific points
δ = EvaluationFunctional([0.0, 0.5, 1.0])
∂x = PartialDerivative((1,))
δ_dx = δ ∘ ∂x

# Evaluate the second derivative
δ_d2x = δ ∘ ∂x ∘ ∂x

# Use in GP conditioning
obs = LinearObservation(δ_dx, derivative_values, noise)
```

# See also
- [`EvaluationFunctional`](@ref): Common base functional
- [`PartialDerivative`](@ref): Common operator for differentiation
"""
struct LinFctlLinFuncOpConcat{N} <: AbstractLinFctlLinFuncOpConcat{N}
    linfctl::AbstractLinearFunctional
    linfuncops::NTuple{N, AbstractLinearFunctionOperator}

    function LinFctlLinFuncOpConcat(
            linfctl::AbstractLinearFunctional,
            linfuncops::NTuple{N, AbstractLinearFunctionOperator},
        ) where {N}
        return new{N}(linfctl, linfuncops)
    end
end

function Base.:∘(op1::AbstractLinearFunctional, op2::AbstractLinearFunctionOperator)
    return LinFctlLinFuncOpConcat(op1, (op2,))
end

function Base.:∘(
        op1::AbstractLinearFunctional,
        op2::AbstractConcatenatedLinearFunctionOperator,
    )
    return LinFctlLinFuncOpConcat(op1, linfuncops(op2))
end

function Base.:∘(op1::AbstractLinFctlLinFuncOpConcat, op2::AbstractLinearFunctionOperator)
    return LinFctlLinFuncOpConcat(linfctl(op1), (op2, linfuncops(op1)...))
end

function Base.:∘(
        op1::AbstractLinFctlLinFuncOpConcat,
        op2::AbstractConcatenatedLinearFunctionOperator,
    )
    return LinFctlLinFuncOpConcat(linfctl(op1), (linfuncops(op2)..., linfuncops(op1)...))
end
