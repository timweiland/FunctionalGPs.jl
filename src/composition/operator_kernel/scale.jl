function (op::AbstractScaledLinearFunctionOperator)(x::T, args...; kwargs...) where {T}
    return scale(op) * linfuncop(op)(x, args...; kwargs...)
end
function _fallback(op::AbstractScaledLinearFunctionOperator, x, args...; kwargs...)
    return invoke(op, Tuple{Any}, x, args...; kwargs...)
end
function (op::AbstractScaledLinearFunctionOperator)(
        x::EvaluationPVCrosscov,
        args...;
        kwargs...,
    )
    return _fallback(op, x, args...; kwargs...)
end
function (op::AbstractScaledLinearFunctionOperator)(
        x::StackedPVCrosscov,
        args...;
        kwargs...,
    )
    return _fallback(op, x, args...; kwargs...)
end
function (op::AbstractScaledLinearFunctionOperator)(
        x::AbstractSumPVCrosscov,
        args...;
        kwargs...,
    )
    return _fallback(op, x, args...; kwargs...)
end
function (op::AbstractScaledLinearFunctionOperator)(
        x::ConstantScaledPVCrosscov,
        args...;
        kwargs...,
    )
    return _fallback(op, x, args...; kwargs...)
end
function (op::AbstractScaledLinearFunctionOperator)(
        x::ZeroMean{T},
        args...;
        kwargs...,
    ) where {T}
    return ZeroMean{T}()
end
function (op::AbstractScaledLinearFunctionOperator)(x::KernelSum, args...; kwargs...)
    return _fallback(op, x, args...; kwargs...)
end
function (op::AbstractScaledLinearFunctionOperator)(x::ScaledKernel, args...; kwargs...)
    return _fallback(op, x, args...; kwargs...)
end
function (op::AbstractScaledLinearFunctionOperator)(
        x::LinearlyScaledKernel,
        args...;
        kwargs...,
    )
    return _fallback(op, x, args...; kwargs...)
end

# Helper: use ScaledKernel for positive scales, LinearlyScaledKernel otherwise.
# s may be a Number or a 1-element container (ScaledKernel stores σ² that way).
_all_positive(s::Number) = s > 0
_all_positive(s) = all(x -> x > 0, s)

function _scaled_kernel(k::Kernel, s)
    if _all_positive(s)
        return ScaledKernel(k, s)
    else
        return LinearlyScaledKernel(k, s)
    end
end

# ConstantScaledLinearFunctionOperator on kernels: use _scaled_kernel to allow
# negative scalars (ScaledKernel requires σ² > 0).
function (op::ConstantScaledLinearFunctionOperator)(k::Kernel, args...; kwargs...)
    return _scaled_kernel(linfuncop(op)(k, args...; kwargs...), op.scalar)
end
function (op::ConstantScaledLinearFunctionOperator)(k::KernelSum, args...; kwargs...)
    return _scaled_kernel(linfuncop(op)(k, args...; kwargs...), op.scalar)
end
function (op::ConstantScaledLinearFunctionOperator)(k::ScaledKernel, args...; kwargs...)
    return _scaled_kernel(
        linfuncop(op)(k.kernel, args...; kwargs...), op.scalar * k.σ²
    )
end
function (op::ConstantScaledLinearFunctionOperator)(
        k::LinearlyScaledKernel,
        args...;
        kwargs...,
    )
    return _scaled_kernel(
        linfuncop(op)(k.kernel, args...; kwargs...), op.scalar * k.scalar
    )
end

# ConstantScaledLinearFunctionOperator on crosscovs: fold scalars.
# Apply the *unscaled* inner operator and fold the two scalars once — mirroring
# the kernel-side methods above. Using the full `op` here would re-apply
# `op.scalar`, scaling the result by `op.scalar² · pv.scalar` instead of
# `op.scalar · pv.scalar` (the source of negative-definite covariances when a
# negatively-scaled operator was applied to both kernel arguments).
function (op::ConstantScaledLinearFunctionOperator)(
        pv::ConstantScaledPVCrosscov,
        args...;
        kwargs...,
    )
    return ConstantScaledPVCrosscov(
        linfuncop(op)(pv.pv_crosscov, args...; kwargs...), op.scalar * pv.scalar
    )
end
