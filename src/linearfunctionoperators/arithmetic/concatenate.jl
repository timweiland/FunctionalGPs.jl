export AbstractConcatenatedLinearFunctionOperator, ConcatenatedLinearFunctionOperator

abstract type AbstractConcatenatedLinearFunctionOperator{N} <: AbstractLinearFunctionOperator end
linfuncops(op::AbstractConcatenatedLinearFunctionOperator) = op.linfuncops

function Base.show(io::IO, op::AbstractConcatenatedLinearFunctionOperator)
    return print(
        io,
        join(["($(string(linfuncop)))" for linfuncop in reverse(linfuncops(op))], " ∘ "),
    )
end

struct ConcatenatedLinearFunctionOperator{N} <: AbstractConcatenatedLinearFunctionOperator{N}
    linfuncops::NTuple{N, AbstractLinearFunctionOperator}

    function ConcatenatedLinearFunctionOperator(
            linfuncops::NTuple{N, AbstractLinearFunctionOperator},
        ) where {N}
        return new{N}(linfuncops)
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
