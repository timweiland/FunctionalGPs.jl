using AbstractGPs: AbstractGP

function _concat_impl(op::AbstractLinFctlLinFuncOpConcat, x, args...; kwargs...)
    res = x
    for linfuncop in linfuncops(op)
        res = linfuncop(res, args...; kwargs...)
    end
    return linfctl(op)(res, args...; kwargs...)
end

(op::AbstractLinFctlLinFuncOpConcat)(k::Kernel, args...; kwargs...) =
    _concat_impl(op, k, args...; kwargs...)
(op::AbstractLinFctlLinFuncOpConcat)(pv::ProcessVectorCrossCovariance, args...; kwargs...) =
    _concat_impl(op, pv, args...; kwargs...)

function (op::AbstractLinFctlLinFuncOpConcat)(::ZeroMean{T}, args...; kwargs...) where {T}
    return zeros(T, output_shape(op)...)
end

(op::AbstractLinFctlLinFuncOpConcat)(pv::StackedPVCrosscov, args...; kwargs...) =
    _concat_impl(op, pv, args...; kwargs...)
(op::AbstractLinFctlLinFuncOpConcat)(pv::EvaluationPVCrosscov, args...; kwargs...) =
    _concat_impl(op, pv, args...; kwargs...)
(op::AbstractLinFctlLinFuncOpConcat)(x::AbstractSumPVCrosscov, args...; kwargs...) =
    _concat_impl(op, x, args...; kwargs...)
(op::AbstractLinFctlLinFuncOpConcat)(x::ConstantScaledPVCrosscov, args...; kwargs...) =
    _concat_impl(op, x, args...; kwargs...)
(op::AbstractLinFctlLinFuncOpConcat)(k::KernelSum, args...; kwargs...) =
    _concat_impl(op, k, args...; kwargs...)
(op::AbstractLinFctlLinFuncOpConcat)(k::ScaledKernel, args...; kwargs...) =
    _concat_impl(op, k, args...; kwargs...)
