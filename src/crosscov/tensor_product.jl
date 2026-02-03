import KernelFunctions: kernelmatrix, ⊗

export AbstractTensorProductCrosscov, factors, kernelmatrix, TensorProductCrosscov, ⊗

"""
    AbstractTensorProductCrosscov <: ProcessVectorCrossCovariance

Abstract type for tensor product cross-covariances.

Represents `pv1 ⊗ pv2 ⊗ ...` where each factor is a [`ProcessVectorCrossCovariance`](@ref).
Tensor products are useful for multi-dimensional domains with separable structure.

# Interface

Subtypes must implement:
- [`factors`](@ref): Return the tuple of factor crosscovs

# See also
- [`TensorProductCrosscov`](@ref): Concrete implementation
- [`FactorizedGrid`](@ref): Grid structure for efficient tensor product evaluation
"""
abstract type AbstractTensorProductCrosscov <: ProcessVectorCrossCovariance end

"""
    factors(op::AbstractTensorProductCrosscov)

Return the factor cross-covariances as a tuple.

# Examples
```julia
julia> tensor_pv = pv_x ⊗ pv_y;
julia> length(factors(tensor_pv))
2
```
"""
factors(op::AbstractTensorProductCrosscov) = op.factors

(op::AbstractTensorProductCrosscov)(args...) = mapreduce(x -> x(args...), *, factors(op))

function Base.show(io::IO, op::AbstractTensorProductCrosscov)
    return print(io, join(["$(string(factor))" for factor in factors(op)], " ⊗ "))
end

function kernelmatrix(op::AbstractTensorProductCrosscov, x::AbstractVector)
    # Generic vector input not implemented yet
    throw(MethodError(kernelmatrix, (op, x)))
end

function kernelmatrix(op::AbstractTensorProductCrosscov, x::FactorizedGrid)
    return mapreduce(
        args -> ((i, factor) = args; kernelmatrix(factor, x[i])),
        kronecker,
        factors(op) |> enumerate |> collect |> reverse,
    )
end

"""
    TensorProductCrosscov{N} <: AbstractTensorProductCrosscov

A cross-covariance that is the tensor (Kronecker) product of `N` other cross-covariances.

Created using the `⊗` operator. All factors must have the same `randvar_arg`.
Tensor products enable efficient computation on structured grids via Kronecker
products of smaller matrices.

# Type Parameters
- `N`: Number of factors

# Fields
- `factors::NTuple{N, ProcessVectorCrossCovariance}`: The factor crosscovs

# Examples
```julia
julia> k = SqExponentialKernel();
julia> X_x = [0.0, 0.5, 1.0];
julia> X_y = [0.0, 1.0];
julia> pv_x = EvaluationFunctional(X_x)(k);
julia> pv_y = EvaluationFunctional(X_y)(k);
julia> tensor_pv = pv_x ⊗ pv_y;
julia> typeof(tensor_pv)
TensorProductCrosscov{2}
julia> randvar_length(tensor_pv)  # 3 * 2 = 6
6
```

# Notes
Use [`FactorizedGrid`](@ref) for efficient kernel matrix computation with tensor
product crosscovs.

# See also
- [`factors`](@ref): Extract the factor crosscovs
- [`⊗`](@ref): Tensor product operator
"""
struct TensorProductCrosscov{N} <: AbstractTensorProductCrosscov
    factors::NTuple{N, ProcessVectorCrossCovariance}

    function TensorProductCrosscov(factors::NTuple{N, ProcessVectorCrossCovariance}) where {N}
        if !allequal(map(randvar_arg, factors))
            throw(ArgumentError("All factors must have the same randvar arg"))
        end
        return new{N}(factors)
    end
end

randvar_batch_size(op::TensorProductCrosscov) = (prod(randvar_length.(factors(op))),)
randvar_arg(op::TensorProductCrosscov) = randvar_arg(factors(op)[1])

function TensorProductCrosscov(factors...)
    return TensorProductCrosscov(factors)
end

"""
    ⊗(pv1::ProcessVectorCrossCovariance, pv2::ProcessVectorCrossCovariance)

Create a tensor product cross-covariance from two cross-covariances.

The resulting crosscov has `randvar_length` equal to the product of the
input lengths. Can be chained: `pv1 ⊗ pv2 ⊗ pv3`.

# Examples
```julia
julia> tensor_pv = pv_x ⊗ pv_y ⊗ pv_z;
julia> randvar_length(tensor_pv)  # product of individual lengths
```
"""
function ⊗(op1::ProcessVectorCrossCovariance, op2::ProcessVectorCrossCovariance)
    return TensorProductCrosscov((op1, op2))
end

function ⊗(op1::AbstractTensorProductCrosscov, op2::ProcessVectorCrossCovariance)
    return TensorProductCrosscov((factors(op1)..., op2))
end

function Base.isequal(op1::TensorProductCrosscov, op2::TensorProductCrosscov)
    return op1.factors == op2.factors
end

function Base.isapprox(op1::TensorProductCrosscov, op2::TensorProductCrosscov)
    return isapprox(op1.factors, op2.factors)
end
