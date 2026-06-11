# ScaledLinearFunctional applied to PVCrosscovs → creates scaled matrices

function _scale_crosscov_impl(op::ScaledLinearFunctional, x, args...; kwargs...)
    return op.scalar * op.linfctl(x, args...; kwargs...)
end

(op::ScaledLinearFunctional)(pv::ProcessVectorCrossCovariance, args...; kwargs...) =
    _scale_crosscov_impl(op, pv, args...; kwargs...)
(op::ScaledLinearFunctional)(pv::EvaluationPVCrosscov, args...; kwargs...) =
    _scale_crosscov_impl(op, pv, args...; kwargs...)
(op::ScaledLinearFunctional)(pv::StackedPVCrosscov, args...; kwargs...) =
    _scale_crosscov_impl(op, pv, args...; kwargs...)
(op::ScaledLinearFunctional)(pv::AbstractSumPVCrosscov, args...; kwargs...) =
    _scale_crosscov_impl(op, pv, args...; kwargs...)
(op::ScaledLinearFunctional)(pv::ConstantScaledPVCrosscov, args...; kwargs...) =
    _scale_crosscov_impl(op, pv, args...; kwargs...)
