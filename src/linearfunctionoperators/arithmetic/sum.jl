export AbstractSumLinearFunctionOperator

abstract type AbstractSumLinearFunctionOperator{N} <: AbstractLinearFunctionOperator end
summands(op::AbstractSumLinearFunctionOperator) = op.summands

function Base.show(io::IO, op::AbstractSumLinearFunctionOperator)
    return print(io, join(["($(string(summand)))" for summand in summands(op)], " + "))
end


struct SumLinearFunctionOperator{N} <: AbstractSumLinearFunctionOperator{N}
    summands::NTuple{N, AbstractLinearFunctionOperator}

    function SumLinearFunctionOperator(summands::NTuple{N, AbstractLinearFunctionOperator}) where {N}
        return new{N}(summands)
    end
end

function Base.:+(op1::AbstractLinearFunctionOperator, op2::AbstractLinearFunctionOperator)
    return SumLinearFunctionOperator((op1, op2))
end

function Base.:+(op1::AbstractSumLinearFunctionOperator, op2::AbstractLinearFunctionOperator)
    return SumLinearFunctionOperator((summands(op1)..., op2))
end

function Base.:+(op1::AbstractLinearFunctionOperator, op2::AbstractSumLinearFunctionOperator)
    return SumLinearFunctionOperator((op1, summands(op2)...))
end

function Base.:+(op1::AbstractSumLinearFunctionOperator, op2::AbstractSumLinearFunctionOperator)
    return SumLinearFunctionOperator((summands(op1)..., summands(op2)...))
end
