export AbstractSumLinearFunctional

abstract type AbstractSumLinearFunctional{N} <: AbstractLinearFunctional end
summands(op::AbstractSumLinearFunctional) = op.summands

function Base.show(io::IO, op::AbstractSumLinearFunctional)
    return print(io, join(["($(string(summand)))" for summand in summands(op)], " + "))
end

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
