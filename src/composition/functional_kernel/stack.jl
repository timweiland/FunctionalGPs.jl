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
    pv_crosscovs = [lf(k, arg = arg) for lf in stacked.linfunctionals]
    return StackedPVCrosscov(pv_crosscovs)
end
