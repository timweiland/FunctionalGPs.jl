# LinFctlLinFuncOpConcat applied to Kernels → creates PVCrosscov

function _concat_kernel_impl(op::AbstractLinFctlLinFuncOpConcat, x, args...; kwargs...)
    res = x
    for linfuncop in linfuncops(op)
        res = linfuncop(res, args...; kwargs...)
    end
    return linfctl(op)(res, args...; kwargs...)
end

(op::AbstractLinFctlLinFuncOpConcat)(k::Kernel, args...; kwargs...) =
    _concat_kernel_impl(op, k, args...; kwargs...)

(op::AbstractLinFctlLinFuncOpConcat)(k::KernelSum, args...; kwargs...) =
    _concat_kernel_impl(op, k, args...; kwargs...)
(op::AbstractLinFctlLinFuncOpConcat)(k::ScaledKernel, args...; kwargs...) =
    _concat_kernel_impl(op, k, args...; kwargs...)
(op::AbstractLinFctlLinFuncOpConcat)(k::LinearlyScaledKernel, args...; kwargs...) =
    k.scalar * _concat_kernel_impl(op, k.kernel, args...; kwargs...)
