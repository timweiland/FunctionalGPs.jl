import KernelFunctions: kernelmatrix

export AbstractSumPVCrosscov, summands, SumPVCrosscov

abstract type AbstractSumPVCrosscov <: ProcessVectorCrossCovariance end
summands(op::AbstractSumPVCrosscov) = op.summands

function (op::AbstractSumPVCrosscov)(args...)
    return sum([summand(args...) for summand in summands(op)])
end

function Base.show(io::IO, op::AbstractSumPVCrosscov)
    return print(io, join(["$(string(summand))" for summand in summands(op)], " + "))
end

function kernelmatrix(op::AbstractSumPVCrosscov, x::AbstractArray)
    return sum([kernelmatrix(summand, x) for summand in summands(op)])
end

struct SumPVCrosscov{N} <: AbstractSumPVCrosscov
    summands::NTuple{N,ProcessVectorCrossCovariance}

    function SumPVCrosscov(summands::NTuple{N,ProcessVectorCrossCovariance}) where {N}
        if !allequal(map(randvar_batch_size, summands))
            throw(ArgumentError("All summands must have the same randvar batch size"))
        end
        if !allequal(map(randvar_arg, summands))
            throw(ArgumentError("All summands must have the same randvar arg"))
        end
        new{N}(summands)
    end
end

function Base.:+(op1::ProcessVectorCrossCovariance, op2::ProcessVectorCrossCovariance)
    return SumPVCrosscov((op1, op2))
end

function Base.:+(op1::AbstractSumPVCrosscov, op2::ProcessVectorCrossCovariance)
    return SumPVCrosscov((summands(op1)..., op2))
end

function Base.:+(op1::ProcessVectorCrossCovariance, op2::AbstractSumPVCrosscov)
    return SumPVCrosscov((op1, summands(op2)...))
end

function Base.:+(op1::AbstractSumPVCrosscov, op2::AbstractSumPVCrosscov)
    return SumPVCrosscov((summands(op1)..., summands(op2)...))
end
