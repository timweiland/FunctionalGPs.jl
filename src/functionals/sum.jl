export AbstractSumLinearFunctional

"""
    AbstractSumLinearFunctional{N} <: AbstractLinearFunctional

Abstract type for sums of linear functionals.

Sums are created using the `+` operator on functionals. All summands must have
the same output shape.

# See also
- [`SumLinearFunctional`](@ref): Concrete implementation
"""
abstract type AbstractSumLinearFunctional{N} <: AbstractLinearFunctional end

"""
    summands(ℒ::AbstractSumLinearFunctional)

Return the tuple of summand functionals.
"""
summands(op::AbstractSumLinearFunctional) = op.summands

function Base.show(io::IO, op::AbstractSumLinearFunctional)
    return print(io, join(["($(string(summand)))" for summand in summands(op)], " + "))
end

"""
    SumLinearFunctional{N} <: AbstractSumLinearFunctional{N}

A sum of N linear functionals, typically created using the `+` operator.

The sum functional applies each summand and adds the results. All summands must
have the same output shape.

# Fields
- `summands::NTuple{N, AbstractLinearFunctional}`: The functionals being summed

# Example
```julia
δ1 = EvaluationFunctional([0.0, 0.5])
δ2 = EvaluationFunctional([1.0, 1.5])
sum_fctl = δ1 + δ2  # Creates SumLinearFunctional
```
"""
struct SumLinearFunctional{N} <: AbstractSumLinearFunctional{N}
    summands::NTuple{N, AbstractLinearFunctional}

    function SumLinearFunctional(summands::NTuple{N, AbstractLinearFunctional}) where {N}
        if !allequal(map(output_shape, summands))
            throw(ArgumentError("All summands must have the same output shape"))
        end
        return new{N}(summands)
    end
end

function Base.:+(op1::AbstractLinearFunctional, op2::AbstractLinearFunctional)
    return SumLinearFunctional((op1, op2))
end

function Base.:+(op1::AbstractSumLinearFunctional, op2::AbstractLinearFunctional)
    return SumLinearFunctional((summands(op1)..., op2))
end

function Base.:+(op1::AbstractLinearFunctional, op2::AbstractSumLinearFunctional)
    return SumLinearFunctional((op1, summands(op2)...))
end

function Base.:+(op1::AbstractSumLinearFunctional, op2::AbstractSumLinearFunctional)
    return SumLinearFunctional((summands(op1)..., summands(op2)...))
end
