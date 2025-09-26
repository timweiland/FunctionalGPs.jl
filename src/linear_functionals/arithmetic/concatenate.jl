using AbstractGPs: AbstractGP

export AbstractLinFctlLinFuncOpConcat, LinFctlLinFuncOpConcat

abstract type AbstractLinFctlLinFuncOpConcat{N} <: AbstractLinearFunctional end
linfctl(op::AbstractLinFctlLinFuncOpConcat) = op.linfctl
linfuncops(op::AbstractLinFctlLinFuncOpConcat) = op.linfuncops
output_shape(op::AbstractLinFctlLinFuncOpConcat) = output_shape(linfctl(op))
function (op::AbstractLinFctlLinFuncOpConcat)(x, args...; kwargs...)
    res = x
    for linfuncop in linfuncops(op)
        res = linfuncop(res, args...; kwargs...)
    end
    return linfctl(op)(res, args...; kwargs...)
end
function _fallback(op::AbstractLinFctlLinFuncOpConcat, x, args...; kwargs...)
    return invoke(op, Tuple{Any}, x, args...; kwargs...)
end
function (op::AbstractLinFctlLinFuncOpConcat)(::ZeroMean{T}, args...; kwargs...) where {T}
    return zeros(T, output_shape(op)...)
end
(op::AbstractLinFctlLinFuncOpConcat)(pv::StackedPVCrosscov, args...; kwargs...) = _fallback(op, pv, args...; kwargs...)
(op::AbstractLinFctlLinFuncOpConcat)(pv::EvaluationPVCrosscov, args...; kwargs...) = _fallback(op, pv, args...; kwargs...)
(op::AbstractLinFctlLinFuncOpConcat)(x::AbstractSumPVCrosscov, args...; kwargs...) = _fallback(op, x, args...; kwargs...)
(op::AbstractLinFctlLinFuncOpConcat)(x::ConstantScaledPVCrosscov, args...; kwargs...) = _fallback(op, x, args...; kwargs...)
(op::AbstractLinFctlLinFuncOpConcat)(k::KernelSum, args...; kwargs...) = _fallback(op, k, args...; kwargs...)
(op::AbstractLinFctlLinFuncOpConcat)(k::ScaledKernel, args...; kwargs...) = _fallback(op, k, args...; kwargs...)

function Base.show(io::IO, op::AbstractLinFctlLinFuncOpConcat)
    return print(
        io,
        "$(string(linfctl(op))) ∘ " *
            join(["($(string(linfuncop)))" for linfuncop in reverse(linfuncops(op))], " ∘ "),
    )
end

struct LinFctlLinFuncOpConcat{N} <: AbstractLinFctlLinFuncOpConcat{N}
    linfctl::AbstractLinearFunctional
    linfuncops::NTuple{N, AbstractLinearFunctionOperator}

    function LinFctlLinFuncOpConcat(
            linfctl::AbstractLinearFunctional,
            linfuncops::NTuple{N, AbstractLinearFunctionOperator},
        ) where {N}
        return new{N}(linfctl, linfuncops)
    end
end

function Base.:∘(op1::AbstractLinearFunctional, op2::AbstractLinearFunctionOperator)
    return LinFctlLinFuncOpConcat(op1, (op2,))
end

function Base.:∘(
        op1::AbstractLinearFunctional,
        op2::AbstractConcatenatedLinearFunctionOperator,
    )
    return LinFctlLinFuncOpConcat(op1, linfuncops(op2))
end

function Base.:∘(op1::AbstractLinFctlLinFuncOpConcat, op2::AbstractLinearFunctionOperator)
    return LinFctlLinFuncOpConcat(linfctl(op1), (op2, linfuncops(op1)...))
end

function Base.:∘(
        op1::AbstractLinFctlLinFuncOpConcat,
        op2::AbstractConcatenatedLinearFunctionOperator,
    )
    return LinFctlLinFuncOpConcat(linfctl(op1), (linfuncops(op2)..., linfuncops(op1)...))
end
