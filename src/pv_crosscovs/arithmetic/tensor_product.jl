import KernelFunctions: kernelmatrix, ⊗

export AbstractTensorProductCrosscov, factors, kernelmatrix, TensorProductCrosscov, ⊗

abstract type AbstractTensorProductCrosscov <: ProcessVectorCrossCovariance end
factors(op::AbstractTensorProductCrosscov) = op.factors

(op::AbstractTensorProductCrosscov)(args...) = mapreduce(x -> x(args...), *, factors(op))

function Base.show(io::IO, op::AbstractTensorProductCrosscov)
    return print(io, join(["$(string(factor))" for factor in factors(op)], " ⊗ "))
end

function kernelmatrix(op::AbstractTensorProductCrosscov, x::AbstractVector)
    # Khatri-rao
    error("Not implemented")
end

function kernelmatrix(op::AbstractTensorProductCrosscov, x::FactorizedGrid)
    return mapreduce(
        args -> ((i, factor) = args; kernelmatrix(factor, x[i])),
        kron,
        factors(op) |> enumerate |> collect |> reverse,
    )
end

struct TensorProductCrosscov{N} <: AbstractTensorProductCrosscov
    factors::NTuple{N,ProcessVectorCrossCovariance}

    function TensorProductCrosscov(factors::NTuple{N,ProcessVectorCrossCovariance}) where {N}
        if !allequal(map(randvar_arg, factors))
            throw(ArgumentError("All factors must have the same randvar arg"))
        end
        new{N}(factors)
    end
end

randvar_batch_size(op::TensorProductCrosscov) = (prod(randvar_length.(factors(op))),)
randvar_arg(op::TensorProductCrosscov) = randvar_arg(factors(op)[1])

function TensorProductCrosscov(factors...)
    return TensorProductCrosscov(factors)
end

function ⊗(op1::ProcessVectorCrossCovariance, op2::ProcessVectorCrossCovariance)
    return TensorProductCrosscov((op1, op2))
end

function Base.isequal(op1::TensorProductCrosscov, op2::TensorProductCrosscov)
    return op1.factors == op2.factors
end

function Base.isapprox(op1::TensorProductCrosscov, op2::TensorProductCrosscov)
    return isapprox(op1.factors, op2.factors)
end

