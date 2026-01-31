# EvaluationFunctional applied to Kernels → creates EvaluationPVCrosscov

function (op::EvaluationFunctional)(k::Kernel; arg::Integer = 2)
    @assert arg ∈ [1, 2]
    return EvaluationPVCrosscov(k, op, arg)
end

(op::EvaluationFunctional)(k::KernelSum, args...; kwargs...) =
    mapreduce((k) -> op(k, args...; kwargs...), +, k.kernels)

(op::EvaluationFunctional)(k::ScaledKernel, args...; kwargs...) =
    k.σ² * op(k.kernel, args...; kwargs...)
