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

function (op::ConstantScaledLinearFunctionOperator)(k::ScaledKernel, args...; kwargs...)
    return ScaledKernel(op.linfuncop(k.kernel, args...; kwargs...), op.scalar * k.σ²)
end
function (op::ConstantScaledLinearFunctionOperator)(pv::ConstantScaledPVCrosscov, args...; kwargs...)
    return ConstantScaledPVCrosscov(op(pv.pv_crosscov, args...; kwargs...), op.scalar * pv.scalar)
end
