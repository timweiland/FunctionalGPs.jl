using KernelFunctions
import KernelFunctions: KernelSum

########### Kernels ###########
function (op::PartialDerivative{1, M})(k::KernelTensorProduct; kwargs...) where {M}
    @assert M == length(op.multi_idx)
    ks = Vector{Kernel}(undef, length(op.multi_idx))
    for (i, order) in enumerate(op.multi_idx)
        pd = PartialDerivative{1, 1}(1, (order,))
        ks[i] = pd(k.kernels[i]; kwargs...)
    end
    return KernelTensorProduct(ks)
end

function (op::PartialDerivative{N, M})(k::KernelSum; kwargs...) where {N, M}
    return KernelSum(map(kernel -> op(kernel; kwargs...), k.kernels))
end

function (op::PartialDerivative{1, 1})(k::KernelSum; kwargs...)
    return KernelSum(map(kernel -> op(kernel; kwargs...), k.kernels))
end

function (op::PartialDerivative{1, 1})(
        k::Kernel;
        arg::Integer = 2,
    )
    if arg ∉ [1, 2]
        throw(DomainError(arg, "arg must be 1 or 2"))
    end
    if arg == 1
        return derivative(k, op.order, 0)
    else
        return derivative(k, 0, op.order)
    end
end

########### PV Crosscovs ###########
function (op::PartialDerivative{1, M})(pv::TensorProductCrosscov{M}) where {M}
    factors = Vector{ProcessVectorCrossCovariance}(undef, M)
    for (i, order) in enumerate(op.multi_idx)
        pd = PartialDerivative((order,))
        factors[i] = pd(pv.factors[i])
    end
    return TensorProductCrosscov(factors...)
end
