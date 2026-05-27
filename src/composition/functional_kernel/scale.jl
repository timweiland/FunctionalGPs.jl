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
