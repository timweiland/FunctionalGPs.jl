using AbstractGPs: AbstractGP

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
(op::AbstractConcatenatedLinearFunctionOperator)(x::EvaluationPVCrosscov, args...; kwargs...) = _fallback(op, x, args...; kwargs...)
(op::AbstractConcatenatedLinearFunctionOperator)(x::StackedPVCrosscov, args...; kwargs...) = _fallback(op, x, args...; kwargs...)
(op::AbstractConcatenatedLinearFunctionOperator)(x::AbstractSumPVCrosscov, args...; kwargs...) = _fallback(op, x, args...; kwargs...)
(op::AbstractConcatenatedLinearFunctionOperator)(x::ConstantScaledPVCrosscov, args...; kwargs...) = _fallback(op, x, args...; kwargs...)
(op::AbstractConcatenatedLinearFunctionOperator)(x::ZeroMean{T}, args...; kwargs...) where {T} = _fallback(op, x, args...; kwargs...)
(op::AbstractConcatenatedLinearFunctionOperator)(x::KernelSum, args...; kwargs...) = _fallback(op, x, args...; kwargs...)
(op::AbstractConcatenatedLinearFunctionOperator)(x::ScaledKernel, args...; kwargs...) = _fallback(op, x, args...; kwargs...)
