export AbstractScaledLinearFunctionOperator, ConstantScaledLinearFunctionOperator

abstract type AbstractScaledLinearFunctionOperator{T} <: AbstractLinearFunctionOperator end
linfuncop(op::AbstractScaledLinearFunctionOperator) = op.linfuncop
function scale(op::AbstractScaledLinearFunctionOperator)
    return error("scale not implemented for $(typeof(op))")
end
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

# struct VariablyScaledLinearFunctionOperator <: AbstractScaledLinearFunctionOperator
#     linfuncop::AbstractLinearFunctionOperator
#     scale_fn::Base.Callable
# end
# scale(op::VariablyScaledLinearFunctionOperator) = op.scale_fn

struct ConstantScaledLinearFunctionOperator{T<:AbstractLinearFunctionOperator} <:
       AbstractScaledLinearFunctionOperator{T}
    linfuncop::AbstractLinearFunctionOperator
    scalar::Number
end
scale(op::ConstantScaledLinearFunctionOperator) = op.scalar

function (op::ConstantScaledLinearFunctionOperator)(k::ScaledKernel, args...; kwargs...)
    return ScaledKernel(op.linfuncop(k.kernel, args...; kwargs...), op.scalar * k.σ²)
end
function (op::ConstantScaledLinearFunctionOperator)(pv::ConstantScaledPVCrosscov, args...; kwargs...)
    return ConstantScaledPVCrosscov(op(pv.pv_crosscov, args...; kwargs...), op.scalar * pv.scalar)
end

# function Base.:(*)(x::Base.Callable, y::AbstractLinearFunctionOperator)
#     return VariablyScaledLinearFunctionOperator(y, x)
# end

function Base.:(*)(x::Number, y::T) where {T<:AbstractLinearFunctionOperator}
    if x == 1
        return y
    end
    return ConstantScaledLinearFunctionOperator{T}(y, x)
end

function Base.:(*)(
    x::Number,
    y::ConstantScaledLinearFunctionOperator{T},
) where {T<:AbstractLinearFunctionOperator}
    if x == 1
        return y
    end
    return ConstantScaledLinearFunctionOperator{T}(y.linfuncop, x * y.scalar)
end

function Base.show(io::IO, op::AbstractScaledLinearFunctionOperator)
    return print(io, "$(string(scale(op))) * ($(string(linfuncop(op))))")
end
