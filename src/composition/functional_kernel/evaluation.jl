# EvaluationFunctional applied to Kernels → creates EvaluationPVCrosscov

function (op::EvaluationFunctional)(k::Kernel; arg::Integer = 2)
    @assert arg ∈ [1, 2]
    return EvaluationPVCrosscov(k, op, arg)
end

function (op::EvaluationFunctional)(::ZeroKernel; arg::Integer = 2)
    @assert arg ∈ [1, 2]
    return ZeroPVCrosscov(output_shape(op), arg)
end

(op::EvaluationFunctional)(k::KernelSum, args...; kwargs...) =
    mapreduce((k) -> op(k, args...; kwargs...), +, k.kernels)

(op::EvaluationFunctional)(k::ScaledKernel, args...; kwargs...) =
    k.σ² * op(k.kernel, args...; kwargs...)

(op::EvaluationFunctional)(k::LinearlyScaledKernel; kwargs...) =
    k.scalar * op(k.kernel; kwargs...)

(op::EvaluationFunctional)(sk::TransformedMultiOutputKernel{<:MultiOutputKernel}; arg = 2) =
    _functional_on_transformed(op, sk; arg = arg)
