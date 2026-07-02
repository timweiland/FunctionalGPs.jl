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

function (op::PartialDerivative)(dk::BlockDiagonalKernel; arg::Integer = 2)
    return BlockDiagonalKernel(map(k -> op(k; arg = arg), dk.kernels))
end

# Any derivative of the zero kernel is the zero kernel (e.g. the off-diagonal block
# of an independent multi-output kernel under a derivative observation).
(op::PartialDerivative)(k::ZeroKernel; arg::Integer = 2) = k
(op::PartialDerivative{1, 1})(k::ZeroKernel; arg::Integer = 2) = k

# On a half-pinned kernel a derivative is not applied to every output eagerly; it
# accumulates onto the pinned argument's operator and is evaluated later on the
# resolved single-output block (the `{1, 1}` method disambiguates against the
# `k::Kernel` derivative path below).
(op::PartialDerivative)(tmk::TransformedMultiOutputKernel; arg::Integer = 2) =
    _accumulate_op(op, tmk)
(op::PartialDerivative{1, 1})(tmk::TransformedMultiOutputKernel; arg::Integer = 2) =
    _accumulate_op(op, tmk)

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

# Disambiguation for ScaledKernel
function (op::PartialDerivative{1, 1})(
        k::ScaledKernel;
        arg::Integer = 2,
    )
    return ScaledKernel(op(k.kernel; arg = arg), k.σ²)
end

# Disambiguation for LinearlyScaledKernel
function (op::PartialDerivative{1, 1})(
        k::LinearlyScaledKernel;
        arg::Integer = 2,
    )
    return LinearlyScaledKernel(op(k.kernel; arg = arg), k.scalar)
end

# Disambiguation for BlockDiagonalKernel
function (op::PartialDerivative{1, 1})(dk::BlockDiagonalKernel; arg::Integer = 2)
    return BlockDiagonalKernel(map(k -> op(k; arg = arg), dk.kernels))
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

########### Squared Exponential multi-D support ###########
# SE kernels have tensor product structure, so multi-index derivatives
# decompose into products of 1D derivatives.

using KernelFunctions: SqExponentialKernel, TransformedKernel, ScaleTransform, ARDTransform

# Disambiguation: 1D case uses the generic derivative path
function (op::PartialDerivative{1, 1})(
        k::SqExponentialKernel;
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

function (op::PartialDerivative{1, 1})(
        k::TransformedKernel{<:SqExponentialKernel, <:ScaleTransform};
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

# Multi-D case (M > 1): decompose via tensor product
function (op::PartialDerivative{1, M})(k::SqExponentialKernel; kwargs...) where {M}
    k_tensor = _to_tensor_product(k, M)
    return op(k_tensor; kwargs...)
end

function (op::PartialDerivative{1, M})(
        k::TransformedKernel{<:SqExponentialKernel, <:ScaleTransform};
        kwargs...,
    ) where {M}
    k_tensor = _to_tensor_product(k, M)
    return op(k_tensor; kwargs...)
end

function (op::PartialDerivative{1, M})(
        k::TransformedKernel{<:SqExponentialKernel, <:ARDTransform};
        kwargs...,
    ) where {M}
    k_tensor = _to_tensor_product(k, M)
    return op(k_tensor; kwargs...)
end
