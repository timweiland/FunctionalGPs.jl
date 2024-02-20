import AbstractGPs: ZeroMean

export AbstractLinearFunctionOperator

abstract type AbstractLinearFunctionOperator end

(::AbstractLinearFunctionOperator)(::ZeroMean{T}, args...) where {T} = ZeroMean{T}()
function (ℒ::AbstractLinearFunctionOperator)(pv::EvaluationPVCrosscov)
    return EvaluationPVCrosscov(ℒ(pv.k; arg = randproc_arg(pv)), pv.X, randvar_arg(pv))
end

function (ℒ::AbstractLinearFunctionOperator)(pv::StackedPVCrosscov)
    return StackedPVCrosscov(map(ℒ, pv.pv_crosscovs))
end
