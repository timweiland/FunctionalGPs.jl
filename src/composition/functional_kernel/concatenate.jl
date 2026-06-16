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

# Disambiguate against base.jl's (::AbstractLinearFunctional)(::TransformedMultiOutputKernel{<:MO}).
# Defer the whole concat (operators *and* functional) into a MultiOutputPVCrosscov
# rather than applying its operators to the half-pinned kernel eagerly — so e.g. an
# inner PartialDerivative only acts once the output is pinned and the block is a
# single-output kernel.
(op::AbstractLinFctlLinFuncOpConcat)(k::TransformedMultiOutputKernel{<:MultiOutputKernel}; arg = 2) =
    _functional_on_transformed(op, k; arg = arg)
