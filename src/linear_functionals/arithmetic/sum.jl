export AbstractSumLinearFunctional

abstract type AbstractSumLinearFunctional{N} <: AbstractLinearFunctional end
summands(op::AbstractSumLinearFunctional) = op.summands
function (op::AbstractSumLinearFunctional)(x, args...)
    return sum([summand(x, args...) for summand in summands(op)])
end

function Base.show(io::IO, op::AbstractSumLinearFunctional)
    return print(io, join(["($(string(summand)))" for summand in summands(op)], " + "))
end

_fallback(op::AbstractSumLinearFunctional, x, args...; kwargs...) = invoke(op, Tuple{Any}, x, args...; kwargs...)
(op::AbstractSumLinearFunctional)(x::EvaluationPVCrosscov, args...; kwargs...) = _fallback(op, x, args...; kwargs...)
(op::AbstractSumLinearFunctional)(x::StackedPVCrosscov, args...; kwargs...) = _fallback(op, x, args...; kwargs...)
(op::AbstractSumLinearFunctional)(x::ZeroMean{T}, args...; kwargs...) where {T} = _fallback(op, x, args...; kwargs...)
(op::AbstractSumLinearFunctional)(x::AbstractSumPVCrosscov, args...; kwargs...) = _fallback(op, x, args...; kwargs...)
(op::AbstractSumLinearFunctional)(x::ConstantScaledPVCrosscov, args...; kwargs...) = _fallback(op, x, args...; kwargs...)
function (ℒ::AbstractSumLinearFunctional)(f::AbstractGP; noise::TΣy = 0) where {TΣy}
    return LinfctlTransformedGP(f, ℒ, ℒ(f.mean), ℒ(ℒ(f.kernel)), noise)
end

struct SumLinearFunctional{N} <: AbstractSumLinearFunctional{N}
    summands::NTuple{N,AbstractLinearFunctional}

    function SumLinearFunctional(summands::NTuple{N,AbstractLinearFunctional}) where {N}
        if !allequal(map(output_shape, summands))
            throw(ArgumentError("All summands must have the same output shape"))
        end
        new{N}(summands)
    end
end

function Base.:+(op1::AbstractLinearFunctional, op2::AbstractLinearFunctional)
    return SumLinearFunctional((op1, op2))
end

function Base.:+(op1::AbstractSumLinearFunctional, op2::AbstractLinearFunctional)
    return SumLinearFunctional((summands(op1)..., op2))
end

function Base.:+(op1::AbstractLinearFunctional, op2::AbstractSumLinearFunctional)
    return SumLinearFunctional((op1, summands(op2)...))
end

function Base.:+(op1::AbstractSumLinearFunctional, op2::AbstractSumLinearFunctional)
    return SumLinearFunctional((summands(op1)..., summands(op2)...))
end
