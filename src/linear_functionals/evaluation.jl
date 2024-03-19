export EvaluationFunctional

struct EvaluationFunctional <: AbstractLinearFunctional
    X::AbstractVector
    output_shape::Tuple{Vararg{Integer}}
end

function EvaluationFunctional(X::AbstractVector)
    return EvaluationFunctional(X, size(X))
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

function _fallback(op::EvaluationFunctional, x, args...; kwargs...)
    return invoke(op, Tuple{Any}, x, args...; kwargs...)
end
(op::EvaluationFunctional)(k::KernelSum, args...; kwargs...) = _fallback(op, k, args...; kwargs...)
(op::EvaluationFunctional)(k::ScaledKernel, args...; kwargs...) = _fallback(op, k, args...; kwargs...)
