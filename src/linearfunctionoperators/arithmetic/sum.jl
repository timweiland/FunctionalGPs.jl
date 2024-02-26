import KernelFunctions: KernelSum, ScaledKernel

export AbstractSumLinearFunctionOperator

abstract type AbstractSumLinearFunctionOperator{N} <: AbstractLinearFunctionOperator end
summands(op::AbstractSumLinearFunctionOperator) = op.summands
function (op::AbstractSumLinearFunctionOperator)(args...; kwargs...)
    return sum([summand(args...; kwargs...) for summand in summands(op)])
end

function Base.show(io::IO, op::AbstractSumLinearFunctionOperator)
    return print(io, join(["($(string(summand)))" for summand in summands(op)], " + "))
end

_fallback(op::AbstractSumLinearFunctionOperator, x, args...; kwargs...) = invoke(op, Tuple{Any}, x, args...; kwargs...)
(op::AbstractSumLinearFunctionOperator)(x::EvaluationPVCrosscov, args...; kwargs...) = _fallback(op, x, args...; kwargs...)
(op::AbstractSumLinearFunctionOperator)(x::StackedPVCrosscov, args...; kwargs...) = _fallback(op, x, args...; kwargs...)
(op::AbstractSumLinearFunctionOperator)(x::ZeroMean{T}, args...; kwargs...) where {T} = ZeroMean{T}()
(op::AbstractSumLinearFunctionOperator)(x::KernelSum, args...; kwargs...) = _fallback(op, x, args...; kwargs...)
(op::AbstractSumLinearFunctionOperator)(x::ScaledKernel, args...; kwargs...) = _fallback(op, x, args...; kwargs...)

struct SumLinearFunctionOperator{N} <: AbstractSumLinearFunctionOperator{N}
    summands::NTuple{N,AbstractLinearFunctionOperator}

    function SumLinearFunctionOperator(summands::NTuple{N,AbstractLinearFunctionOperator}) where {N}
        new{N}(summands)
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
