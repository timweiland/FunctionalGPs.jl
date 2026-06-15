# ScaledLinearFunctional applied to Kernels → creates scaled PVCrosscov

function _scale_kernel_impl(op::ScaledLinearFunctional, x, args...; kwargs...)
    return op.scalar * op.linfctl(x, args...; kwargs...)
end

(op::ScaledLinearFunctional)(k::Kernel, args...; kwargs...) =
    _scale_kernel_impl(op, k, args...; kwargs...)
(op::ScaledLinearFunctional)(k::KernelSum, args...; kwargs...) =
    _scale_kernel_impl(op, k, args...; kwargs...)
(op::ScaledLinearFunctional)(k::ScaledKernel, args...; kwargs...) =
    _scale_kernel_impl(op, k, args...; kwargs...)
(op::ScaledLinearFunctional)(k::LinearlyScaledKernel, args...; kwargs...) =
    _scale_kernel_impl(op, k, args...; kwargs...)

# Disambiguate half-pinned multi-output SelectedKernels against the generic
# AbstractLinearFunctional methods in base.jl and the k::Kernel method above. The
# inner functional resolves the SelectedKernel; scaling its result wraps the
# MultiOutputPVCrosscov in a ConstantScaledPVCrosscov.
(op::ScaledLinearFunctional)(
    sk::SelectedKernel{<:MultiOutputKernel, Nothing, <:Integer}, args...; kwargs...,
) = _scale_kernel_impl(op, sk, args...; kwargs...)
(op::ScaledLinearFunctional)(
    sk::SelectedKernel{<:MultiOutputKernel, <:Integer, Nothing}, args...; kwargs...,
) = _scale_kernel_impl(op, sk, args...; kwargs...)
