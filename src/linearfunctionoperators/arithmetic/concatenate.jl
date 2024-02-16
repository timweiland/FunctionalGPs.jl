export AbstractConcatenatedLinearFunctionOperator, ConcatenatedLinearFunctionOperator

abstract type AbstractConcatenatedLinearFunctionOperator <: AbstractLinearFunctionOperator end
linfuncops(op::AbstractConcatenatedLinearFunctionOperator) = op.linfuncops
function (op::AbstractConcatenatedLinearFunctionOperator)(x, args...)
    res = x
    for linfuncop in linfuncops(op)
        res = linfuncop(res, args...)
    end
    return res
end

function Base.show(io::IO, op::AbstractConcatenatedLinearFunctionOperator)
    return print(
        io,
        join(["($(string(linfuncop)))" for linfuncop in reverse(linfuncops(op))], " ∘ "),
    )
end

struct ConcatenatedLinearFunctionOperator{N} <: AbstractConcatenatedLinearFunctionOperator
    linfuncops::NTuple{N,AbstractLinearFunctionOperator}
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
