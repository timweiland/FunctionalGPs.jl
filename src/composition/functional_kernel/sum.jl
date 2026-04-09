# SumLinearFunctional applied to Kernels → creates sum of PVCrosscovs

function _sum_kernel_impl(op::AbstractSumLinearFunctional, x, args...; kwargs...)
    return sum([summand(x, args...; kwargs...) for summand in summands(op)])
end

function (op::AbstractSumLinearFunctional)(k::Kernel, args...; kwargs...)
    return _sum_kernel_impl(op, k, args...; kwargs...)
end

(op::AbstractSumLinearFunctional)(k::KernelSum, args...; kwargs...) =
    _sum_kernel_impl(op, k, args...; kwargs...)
(op::AbstractSumLinearFunctional)(k::ScaledKernel, args...; kwargs...) =
    _sum_kernel_impl(op, k, args...; kwargs...)
(op::AbstractSumLinearFunctional)(k::LinearlyScaledKernel, args...; kwargs...) =
    k.scalar * _sum_kernel_impl(op, k.kernel, args...; kwargs...)
