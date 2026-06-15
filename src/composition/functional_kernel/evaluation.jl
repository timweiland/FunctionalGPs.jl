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

# Disambiguate half-pinned multi-output SelectedKernels against the generic
# AbstractLinearFunctional methods in base.jl and the k::Kernel method above.
(op::EvaluationFunctional)(
    sk::SelectedKernel{<:MultiOutputKernel, Nothing, <:Integer};
    arg = 2,
) = (
    (arg == 2)
    ? MultiOutputPVCrosscov{2}(sk.parent, sk.pin2, op)
    : Select(sk.pin2)(op(sk.parent; arg = 1))
)

(op::EvaluationFunctional)(
    sk::SelectedKernel{<:MultiOutputKernel, <:Integer, Nothing};
    arg = 2,
) = (
    (arg == 2)
    ? Select(sk.pin1)(op(sk.parent; arg = 2))
    : MultiOutputPVCrosscov{1}(sk.parent, sk.pin1, op)
)
