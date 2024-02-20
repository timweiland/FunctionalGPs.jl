export AbstractConcatenatedLinearFunctionOperator, ConcatenatedLinearFunctionOperator
using AbstractGPs: AbstractGP

abstract type AbstractConcatenatedLinearFunctionOperator <: AbstractLinearFunctionOperator end
linfuncops(op::AbstractConcatenatedLinearFunctionOperator) = op.linfuncops
function (op::AbstractConcatenatedLinearFunctionOperator)(
    x::T,
    args...;
    kwargs...,
) where {T}
    res = x
    for linfuncop in linfuncops(op)
        res = linfuncop(res, args...; kwargs...)
    end
    return res
end
_fallback(op::AbstractConcatenatedLinearFunctionOperator, x, args...; kwargs...) = invoke(op, Tuple{Any}, x, args...; kwargs...)
(op::AbstractConcatenatedLinearFunctionOperator)(x::EvaluationPVCrosscov) = _fallback(op, x)
(op::AbstractConcatenatedLinearFunctionOperator)(x::StackedPVCrosscov) = _fallback(op, x)
(op::AbstractConcatenatedLinearFunctionOperator)(x::ZeroMean{T}, args...) where {T} = _fallback(op, x, args...)

function (ℒ::AbstractConcatenatedLinearFunctionOperator)(f::AbstractGP; noise::TΣy = 0) where {TΣy}
    return LinfctlTransformedGP(f, ℒ, ℒ(f.mean), ℒ(ℒ(f.kernel)), noise)
end

function Base.show(io::IO, op::AbstractConcatenatedLinearFunctionOperator)
    return print(
        io,
        join(["($(string(linfuncop)))" for linfuncop in reverse(linfuncops(op))], " ∘ "),
    )
end

struct ConcatenatedLinearFunctionOperator{N} <: AbstractConcatenatedLinearFunctionOperator
    linfuncops::NTuple{N,AbstractLinearFunctionOperator}
end

function Base.:∘(op1::AbstractLinearFunctionOperator, op2::AbstractLinearFunctionOperator)
    return ConcatenatedLinearFunctionOperator((op2, op1))
end

function Base.:∘(
    op1::AbstractConcatenatedLinearFunctionOperator,
    op2::AbstractLinearFunctionOperator,
)
    return ConcatenatedLinearFunctionOperator((op2, linfuncops(op1)...))
end
