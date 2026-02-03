import KernelFunctions: kernelmatrix

export AbstractSumPVCrosscov, summands, SumPVCrosscov

"""
    AbstractSumPVCrosscov <: ProcessVectorCrossCovariance

Abstract type for cross-covariances that are sums of other cross-covariances.

Represents `pv1 + pv2 + ...` where each summand is a [`ProcessVectorCrossCovariance`](@ref).

# Interface

Subtypes must implement:
- [`summands`](@ref): Return the tuple of summand crosscovs

# See also
- [`SumPVCrosscov`](@ref): Concrete implementation
"""
abstract type AbstractSumPVCrosscov <: ProcessVectorCrossCovariance end

"""
    summands(op::AbstractSumPVCrosscov)

Return the summand cross-covariances as a tuple.

# Examples
```julia
julia> sum_pv = pv1 + pv2 + pv3;
julia> length(summands(sum_pv))
3
```
"""
summands(op::AbstractSumPVCrosscov) = op.summands

function (op::AbstractSumPVCrosscov)(args...)
    return sum([summand(args...) for summand in summands(op)])
end

function Base.show(io::IO, op::AbstractSumPVCrosscov)
    return print(io, join(["$(string(summand))" for summand in summands(op)], " + "))
end

function kernelmatrix(op::AbstractSumPVCrosscov, x::AbstractVector)
    return sum([kernelmatrix(summand, x) for summand in summands(op)])
end

"""
    SumPVCrosscov{N} <: AbstractSumPVCrosscov

A cross-covariance that is the sum of `N` other cross-covariances.

Created automatically when adding [`ProcessVectorCrossCovariance`](@ref) objects
using the `+` operator. All summands must have the same `randvar_batch_size` and
`randvar_arg`.

# Type Parameters
- `N`: Number of summands

# Fields
- `summands::NTuple{N, ProcessVectorCrossCovariance}`: The summand crosscovs

# Examples
```julia
julia> k1 = SqExponentialKernel();
julia> k2 = Matern32Kernel();
julia> X = [0.0, 0.5, 1.0];
julia> pv1 = EvaluationFunctional(X)(k1);
julia> pv2 = EvaluationFunctional(X)(k2);
julia> sum_pv = pv1 + pv2;
julia> typeof(sum_pv)
SumPVCrosscov{2}
```

# See also
- [`summands`](@ref): Extract the summand crosscovs
"""
struct SumPVCrosscov{N} <: AbstractSumPVCrosscov
    summands::NTuple{N, ProcessVectorCrossCovariance}

    function SumPVCrosscov(summands::NTuple{N, ProcessVectorCrossCovariance}) where {N}
        if !allequal(map(randvar_batch_size, summands))
            throw(ArgumentError("All summands must have the same randvar batch size"))
        end
        if !allequal(map(randvar_arg, summands))
            throw(ArgumentError("All summands must have the same randvar arg"))
        end
        return new{N}(summands)
    end
end

randvar_batch_size(op::SumPVCrosscov) = randvar_batch_size(summands(op)[1])
randvar_arg(op::SumPVCrosscov) = randvar_arg(summands(op)[1])

function Base.:(-)(pv::SumPVCrosscov)
    return SumPVCrosscov(map(-, pv.summands))
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

function Base.isequal(op1::SumPVCrosscov, op2::SumPVCrosscov)
    return all(op1.summands .== op2.summands)
end

function Base.isapprox(op1::SumPVCrosscov, op2::SumPVCrosscov)
    return all(isapprox.(op1.summands, op2.summands))
end
