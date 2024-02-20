export SumLinearFunctionOperator

abstract type SumLinearFunctionOperator <: AbstractLinearFunctionOperator end
summands(op::SumLinearFunctionOperator) = op.summands
function (op::SumLinearFunctionOperator)(args...)
    return sum([summand(args...) for summand in summands(op)])
end

function Base.show(io::IO, op::SumLinearFunctionOperator)
    return print(io, join(["($(string(summand)))" for summand in summands(op)], " + "))
end

_fallback(op::SumLinearFunctionOperator, x, args...; kwargs...) = invoke(op, Tuple{Any}, x, args...; kwargs...)
(op::SumLinearFunctionOperator)(x::EvaluationPVCrosscov) = _fallback(op, x)
(op::SumLinearFunctionOperator)(x::StackedPVCrosscov) = _fallback(op, x)
(op::SumLinearFunctionOperator)(x::ZeroMean{T}, args...) where {T} = _fallback(op, x, args...)
function (ℒ::SumLinearFunctionOperator)(f::AbstractGP; noise::TΣy = 0) where {TΣy}
    return LinfctlTransformedGP(f, ℒ, ℒ(f.mean), ℒ(ℒ(f.kernel)), noise)
end