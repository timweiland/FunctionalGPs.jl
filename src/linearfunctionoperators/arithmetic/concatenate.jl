export AbstractConcatenatedLinearFunctionOperator, ConcatenatedLinearFunctionOperator
using AbstractGPs: AbstractGP

abstract type AbstractConcatenatedLinearFunctionOperator{N} <: AbstractLinearFunctionOperator end
linfuncops(op::AbstractConcatenatedLinearFunctionOperator) = op.linfuncops
function (op::AbstractConcatenatedLinearFunctionOperator)(
    x::T,
    args...;
    kwargs...,
) where {T}
    res = x
    for linfuncop in linfuncops(op)
        res = linfuncop(res, args...; kwargs...)
    end
    return res
end
_fallback(op::AbstractConcatenatedLinearFunctionOperator, x, args...; kwargs...) = invoke(op, Tuple{Any}, x, args...; kwargs...)
(op::AbstractConcatenatedLinearFunctionOperator)(x::EvaluationPVCrosscov, args...; kwargs...) = _fallback(op, x, args...; kwargs...)
(op::AbstractConcatenatedLinearFunctionOperator)(x::StackedPVCrosscov, args...; kwargs...) = _fallback(op, x, args...; kwargs...)
(op::AbstractConcatenatedLinearFunctionOperator)(x::ZeroMean{T}, args...; kwargs...) where {T} = _fallback(op, x, args...; kwargs...)
(op::AbstractConcatenatedLinearFunctionOperator)(x::KernelSum, args...; kwargs...) = _fallback(op, x, args...; kwargs...)
(op::AbstractConcatenatedLinearFunctionOperator)(x::ScaledKernel, args...; kwargs...) = _fallback(op, x, args...; kwargs...)


function Base.show(io::IO, op::AbstractConcatenatedLinearFunctionOperator)
    return print(
        io,
        join(["($(string(linfuncop)))" for linfuncop in reverse(linfuncops(op))], " ∘ "),
    )
end

struct ConcatenatedLinearFunctionOperator{N} <: AbstractConcatenatedLinearFunctionOperator{N}
    linfuncops::NTuple{N,AbstractLinearFunctionOperator}

    function ConcatenatedLinearFunctionOperator(
        linfuncops::NTuple{N,AbstractLinearFunctionOperator},
    ) where {N}
        new{N}(linfuncops)
    end
end

function Base.:∘(op1::AbstractLinearFunctionOperator, op2::AbstractLinearFunctionOperator)
    return ConcatenatedLinearFunctionOperator((op2, op1))
end

function Base.:∘(
    op1::AbstractConcatenatedLinearFunctionOperator,
    op2::AbstractLinearFunctionOperator,
)
    return ConcatenatedLinearFunctionOperator((op2, linfuncops(op1)...))
end

function Base.:∘(
    op1::AbstractLinearFunctionOperator,
    op2::AbstractConcatenatedLinearFunctionOperator,
)
    return ConcatenatedLinearFunctionOperator((linfuncops(op2)..., op1))
end

function Base.:∘(
    op1::AbstractConcatenatedLinearFunctionOperator,
    op2::AbstractConcatenatedLinearFunctionOperator,
)
    return ConcatenatedLinearFunctionOperator((linfuncops(op2)..., linfuncops(op1)...))
end
