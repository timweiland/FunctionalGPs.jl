# Generic AbstractLinearFunctional applied to Kernels

# Any linear functional applied to the zero kernel produces the zero crosscov.
function (ℒ::AbstractLinearFunctional)(::ZeroKernel; arg::Integer = 2)
    argi = Int(arg)
    @assert argi ∈ (1, 2)
    return ZeroPVCrosscov(output_shape(ℒ), argi)
end

# Handle KernelSum - apply to each component
function (ℒ::AbstractLinearFunctional)(k::KernelSum, args...; kwargs...)
    return mapreduce((k) -> ℒ(k, args...; kwargs...), +, k.kernels)
end

# Handle ScaledKernel - scale the result
function (ℒ::AbstractLinearFunctional)(k::ScaledKernel, args...; kwargs...)
    return k.σ² * ℒ(k.kernel, args...; kwargs...)
end

# Handle LinearlyScaledKernel - same pattern, allows negative scalars
function (ℒ::AbstractLinearFunctional)(k::LinearlyScaledKernel, args...; kwargs...)
    return k.scalar * ℒ(k.kernel, args...; kwargs...)
end

# Handle SelectedKernel
(ℒ::AbstractLinearFunctional)(
    sk::SelectedKernel{<:MultiOutputKernel, Nothing, <:Integer};
    arg = 2,
) = (
    (arg == 2)
    ? MultiOutputPVCrosscov{2}(sk.parent, sk.pin2, ℒ)
    : Select(sk.pin2)(ℒ(sk.parent; arg = 1))
)

(ℒ::AbstractLinearFunctional)(
    sk::SelectedKernel{<:MultiOutputKernel, <:Integer, Nothing};
    arg = 2,
) = (
    (arg == 2)
    ? Select(sk.pin1)(ℒ(sk.parent; arg = 2))
    : MultiOutputPVCrosscov{1}(sk.parent, sk.pin1, ℒ)
)
