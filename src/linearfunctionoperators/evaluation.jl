export EvaluationFunctional

struct EvaluationFunctional <: AbstractLinearFunctionOperator
    X::AbstractVector
end

function (op::EvaluationFunctional)(k::Kernel; arg::Integer = 2)
    @assert arg ∈ [1, 2]
    return EvaluationPVCrosscov(k, op.X, arg)
end

function (op::EvaluationFunctional)(pv::EvaluationPVCrosscov)
    X₁ = pv.eval_arg == 1 ? pv.X : op.X
    X₂ = pv.eval_arg == 2 ? pv.X : op.X
    return kernelmatrix(pv.k, X₁, X₂)
end
