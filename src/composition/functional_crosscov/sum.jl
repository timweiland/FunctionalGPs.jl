# SumLinearFunctional applied to PVCrosscovs → creates matrices

function _sum_crosscov_impl(op::AbstractSumLinearFunctional, x, args...; kwargs...)
    return sum([summand(x, args...; kwargs...) for summand in summands(op)])
end

# Apply to generic ProcessVectorCrossCovariance
function (op::AbstractSumLinearFunctional)(pv::ProcessVectorCrossCovariance, args...; kwargs...)
    return _sum_crosscov_impl(op, pv, args...; kwargs...)
end

# Apply to specific crosscov types
(op::AbstractSumLinearFunctional)(x::EvaluationPVCrosscov, args...; kwargs...) =
    _sum_crosscov_impl(op, x, args...; kwargs...)
(op::AbstractSumLinearFunctional)(x::StackedPVCrosscov, args...; kwargs...) =
    _sum_crosscov_impl(op, x, args...; kwargs...)
(op::AbstractSumLinearFunctional)(x::AbstractSumPVCrosscov, args...; kwargs...) =
    _sum_crosscov_impl(op, x, args...; kwargs...)
(op::AbstractSumLinearFunctional)(x::ConstantScaledPVCrosscov, args...; kwargs...) =
    _sum_crosscov_impl(op, x, args...; kwargs...)
