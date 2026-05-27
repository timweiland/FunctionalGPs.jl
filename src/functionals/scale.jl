export ScaledLinearFunctional

struct ScaledLinearFunctional <: AbstractLinearFunctional
    linfctl::AbstractLinearFunctional
    scalar::Number
end

output_shape(op::ScaledLinearFunctional) = output_shape(op.linfctl)

function Base.:(*)(x::Number, y::ScaledLinearFunctional)
    s = x * y.scalar
    if s == 1
        return y.linfctl
    end
    return ScaledLinearFunctional(y.linfctl, s)
end

function Base.:(*)(x::Number, y::AbstractLinearFunctional)
    if x == 1
        return y
    end
    return ScaledLinearFunctional(y, x)
end

function Base.:(-)(op::AbstractLinearFunctional)
    return (-1) * op
end

function Base.:(-)(a::AbstractLinearFunctional, b::AbstractLinearFunctional)
    return a + (-b)
end

function Base.show(io::IO, op::ScaledLinearFunctional)
    return print(io, "$(op.scalar) * ($(string(op.linfctl)))")
end
