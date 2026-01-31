# LinFctlLinFuncOpConcat applied to PVCrosscovs → creates matrices

using AbstractGPs: ZeroMean

function _concat_crosscov_impl(op::AbstractLinFctlLinFuncOpConcat, x, args...; kwargs...)
    res = x
    for linfuncop in linfuncops(op)
        res = linfuncop(res, args...; kwargs...)
    end
    return linfctl(op)(res, args...; kwargs...)
end

# Apply to ZeroMean
function (op::AbstractLinFctlLinFuncOpConcat)(::ZeroMean{T}, args...; kwargs...) where {T}
    return zeros(T, output_shape(op)...)
end

# Apply to generic ProcessVectorCrossCovariance
(op::AbstractLinFctlLinFuncOpConcat)(pv::ProcessVectorCrossCovariance, args...; kwargs...) =
    _concat_crosscov_impl(op, pv, args...; kwargs...)

# Apply to specific crosscov types
(op::AbstractLinFctlLinFuncOpConcat)(pv::StackedPVCrosscov, args...; kwargs...) =
    _concat_crosscov_impl(op, pv, args...; kwargs...)
(op::AbstractLinFctlLinFuncOpConcat)(pv::EvaluationPVCrosscov, args...; kwargs...) =
    _concat_crosscov_impl(op, pv, args...; kwargs...)
(op::AbstractLinFctlLinFuncOpConcat)(x::AbstractSumPVCrosscov, args...; kwargs...) =
    _concat_crosscov_impl(op, x, args...; kwargs...)
(op::AbstractLinFctlLinFuncOpConcat)(x::ConstantScaledPVCrosscov, args...; kwargs...) =
    _concat_crosscov_impl(op, x, args...; kwargs...)
