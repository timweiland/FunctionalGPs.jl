using AbstractGPs

function _sum_impl(op::AbstractSumLinearFunctional, x, args...; kwargs...)
    return sum([summand(x, args...; kwargs...) for summand in summands(op)])
end

function (op::AbstractSumLinearFunctional)(k::Kernel, args...; kwargs...)
    return _sum_impl(op, k, args...; kwargs...)
end

function (op::AbstractSumLinearFunctional)(pv::ProcessVectorCrossCovariance, args...; kwargs...)
    return _sum_impl(op, pv, args...; kwargs...)
end

(op::AbstractSumLinearFunctional)(x::ZeroMean, args...; kwargs...) = _sum_impl(op, x, args...; kwargs...)

(op::AbstractSumLinearFunctional)(x::EvaluationPVCrosscov, args...; kwargs...) = _sum_impl(op, x, args...; kwargs...)
(op::AbstractSumLinearFunctional)(x::StackedPVCrosscov, args...; kwargs...) = _sum_impl(op, x, args...; kwargs...)
(op::AbstractSumLinearFunctional)(x::AbstractSumPVCrosscov, args...; kwargs...) = _sum_impl(op, x, args...; kwargs...)
(op::AbstractSumLinearFunctional)(x::ConstantScaledPVCrosscov, args...; kwargs...) = _sum_impl(op, x, args...; kwargs...)

(op::AbstractSumLinearFunctional)(k::KernelSum, args...; kwargs...) = _sum_impl(op, k, args...; kwargs...)
(op::AbstractSumLinearFunctional)(k::ScaledKernel, args...; kwargs...) = _sum_impl(op, k, args...; kwargs...)
