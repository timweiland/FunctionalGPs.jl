# Generic AbstractLinearFunctional applied to Kernels

# Handle KernelSum - apply to each component
function (ℒ::AbstractLinearFunctional)(k::KernelSum, args...; kwargs...)
    return mapreduce((k) -> ℒ(k, args...; kwargs...), +, k.kernels)
end

# Handle ScaledKernel - scale the result
function (ℒ::AbstractLinearFunctional)(k::ScaledKernel, args...; kwargs...)
    return k.σ² * ℒ(k.kernel, args...; kwargs...)
end
