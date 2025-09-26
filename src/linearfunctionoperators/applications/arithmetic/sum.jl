import KernelFunctions: KernelSum, ScaledKernel

function (op::AbstractSumLinearFunctionOperator)(args...; kwargs...)
    return sum([summand(args...; kwargs...) for summand in summands(op)])
end

_fallback(op::AbstractSumLinearFunctionOperator, x, args...; kwargs...) = invoke(op, Tuple{Any}, x, args...; kwargs...)
(op::AbstractSumLinearFunctionOperator)(x::EvaluationPVCrosscov, args...; kwargs...) = _fallback(op, x, args...; kwargs...)
(op::AbstractSumLinearFunctionOperator)(x::StackedPVCrosscov, args...; kwargs...) = _fallback(op, x, args...; kwargs...)
(op::AbstractSumLinearFunctionOperator)(x::AbstractSumPVCrosscov, args...; kwargs...) = _fallback(op, x, args...; kwargs...)
(op::AbstractSumLinearFunctionOperator)(x::ConstantScaledPVCrosscov, args...; kwargs...) = _fallback(op, x, args...; kwargs...)
(op::AbstractSumLinearFunctionOperator)(x::ZeroMean{T}, args...; kwargs...) where {T} = ZeroMean{T}()
(op::AbstractSumLinearFunctionOperator)(x::KernelSum, args...; kwargs...) = _fallback(op, x, args...; kwargs...)
(op::AbstractSumLinearFunctionOperator)(x::ScaledKernel, args...; kwargs...) = _fallback(op, x, args...; kwargs...)
