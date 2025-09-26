export AbstractLinFctlLinFuncOpConcat, LinFctlLinFuncOpConcat

abstract type AbstractLinFctlLinFuncOpConcat{N} <: AbstractLinearFunctional end
linfctl(op::AbstractLinFctlLinFuncOpConcat) = op.linfctl
linfuncops(op::AbstractLinFctlLinFuncOpConcat) = op.linfuncops
output_shape(op::AbstractLinFctlLinFuncOpConcat) = output_shape(linfctl(op))

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
