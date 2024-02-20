export SumLinearFunctional

abstract type SumLinearFunctional <: AbstractLinearFunctional end
summands(op::SumLinearFunctional) = op.summands
function (op::SumLinearFunctional)(args...)
    return sum([summand(args...) for summand in summands(op)])
end

function Base.show(io::IO, op::SumLinearFunctional)
    return print(io, join(["($(string(summand)))" for summand in summands(op)], " + "))
end

_fallback(op::SumLinearFunctional, x, args...; kwargs...) = invoke(op, Tuple{Any}, x, args...; kwargs...)
(op::SumLinearFunctional)(x::EvaluationPVCrosscov) = _fallback(op, x)
(op::SumLinearFunctional)(x::StackedPVCrosscov) = _fallback(op, x)
(op::SumLinearFunctional)(x::ZeroMean{T}, args...) where {T} = _fallback(op, x, args...)
function (ℒ::SumLinearFunctional)(f::AbstractGP; noise::TΣy = 0) where {TΣy}
    return LinfctlTransformedGP(f, ℒ, ℒ(f.mean), ℒ(ℒ(f.kernel)), noise)
end