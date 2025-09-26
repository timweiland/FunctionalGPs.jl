export AbstractScaledLinearFunctionOperator, ConstantScaledLinearFunctionOperator

abstract type AbstractScaledLinearFunctionOperator{T} <: AbstractLinearFunctionOperator end
linfuncop(op::AbstractScaledLinearFunctionOperator) = op.linfuncop
function scale(op::AbstractScaledLinearFunctionOperator)
    return error("scale not implemented for $(typeof(op))")
end


struct ConstantScaledLinearFunctionOperator{T <: AbstractLinearFunctionOperator} <:
    AbstractScaledLinearFunctionOperator{T}
    linfuncop::AbstractLinearFunctionOperator
    scalar::Number
end
scale(op::ConstantScaledLinearFunctionOperator) = op.scalar

function Base.:(*)(x::Number, y::T) where {T <: AbstractLinearFunctionOperator}
    if x == 1
        return y
    end
    return ConstantScaledLinearFunctionOperator{T}(y, x)
end

function Base.:(*)(
        x::Number,
        y::ConstantScaledLinearFunctionOperator{T},
    ) where {T <: AbstractLinearFunctionOperator}
    if x == 1
        return y
    end
    return ConstantScaledLinearFunctionOperator{T}(y.linfuncop, x * y.scalar)
end

function Base.show(io::IO, op::AbstractScaledLinearFunctionOperator)
    return print(io, "$(string(scale(op))) * ($(string(linfuncop(op))))")
end
