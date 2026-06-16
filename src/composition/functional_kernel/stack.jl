# StackedLinearFunctional applied to Kernels → creates StackedPVCrosscov

"""
    (stacked::StackedLinearFunctional)(k::Kernel; arg=2)

Apply a stacked linear functional to a kernel, creating a StackedPVCrosscov where each
component is the result of applying each individual functional to the kernel.

# Arguments
- `k::Kernel`: The kernel to apply to
- `arg::Int=2`: Which argument of the kernel to apply to (1 or 2)

# Returns
- `StackedPVCrosscov`: A stacked process-vector crosscovariance
"""
function (stacked::StackedLinearFunctional)(k::Kernel; arg::Integer = 2)
    pv_crosscovs = [lf(k; arg = arg) for lf in stacked.linfunctionals]
    return StackedPVCrosscov(pv_crosscovs)
end

# Disambiguation for KernelSum and ScaledKernel (vs AbstractLinearFunctional methods)
(stacked::StackedLinearFunctional)(k::KernelSum; arg::Integer = 2) =
    StackedPVCrosscov([lf(k; arg = arg) for lf in stacked.linfunctionals])
(stacked::StackedLinearFunctional)(k::ScaledKernel; arg::Integer = 2) =
    k.σ² * stacked(k.kernel; arg = arg)

function (stacked::StackedLinearFunctional)(k::LinearlyScaledKernel; arg::Integer = 2)
    return k.scalar * stacked(k.kernel; arg = arg)
end

# Disambiguate against base.jl's (::AbstractLinearFunctional)(::TransformedMultiOutputKernel{<:MO}).
# Defer the whole stacked functional into a MultiOutputPVCrosscov; it resolves to a
# StackedPVCrosscov once the output is pinned and the block is a single-output kernel.
(stacked::StackedLinearFunctional)(k::TransformedMultiOutputKernel{<:MultiOutputKernel}; arg::Integer = 2) =
    _functional_on_transformed(stacked, k; arg = arg)
