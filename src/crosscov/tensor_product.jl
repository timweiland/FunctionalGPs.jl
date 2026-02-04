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

# Vector-of-vectors input for tensor product crosscov
# The matrix orientation depends on randvar_arg:
# - randvar_arg=1: factor matrices are (n_randvar_d × n_points), use column-wise Khatri-Rao
# - randvar_arg=2: factor matrices are (n_points × n_randvar_d), use row-wise Khatri-Rao
function kernelmatrix(op::AbstractTensorProductCrosscov, x::AbstractVector{<:AbstractVector})
    N = length(factors(op))
    # Extract coordinates for each dimension
    coords = ntuple(d -> [xi[d] for xi in x], N)
    # Compute factor matrices
    Ks = [kernelmatrix(factors(op)[d], coords[d]) for d in 1:N]
    # Combine with appropriate Khatri-Rao variant based on randvar_arg
    if randvar_arg(op) == 1
        # Randvars on rows, points on columns: column-wise Khatri-Rao
        return _khatri_rao_columns(Ks)
    else
        # Points on rows, randvars on columns: row-wise Khatri-Rao
        return _khatri_rao_rows(Ks)
    end
end

function kernelmatrix(
        op::AbstractTensorProductCrosscov,
        x::AbstractVector{<:AbstractVector},
        y::AbstractVector{<:AbstractVector},
    )
    N = length(factors(op))
    # Extract coordinates for each dimension
    coords_x = ntuple(d -> [xi[d] for xi in x], N)
    coords_y = ntuple(d -> [yi[d] for yi in y], N)
    # Compute factor matrices
    Ks = [kernelmatrix(factors(op)[d], coords_x[d], coords_y[d]) for d in 1:N]
    # Combine with appropriate Khatri-Rao variant based on randvar_arg
    if randvar_arg(op) == 1
        return _khatri_rao_columns(Ks)
    else
        return _khatri_rao_rows(Ks)
    end
end

# Column-wise Kronecker product (Khatri-Rao product)
# Given matrices K1 (m1 × n), K2 (m2 × n), returns (m1*m2 × n) matrix
# where each column is kron(K2[:, j], K1[:, j])
function _khatri_rao_columns(Ks::Vector)
    n_cols = size(Ks[1], 2)
    # Start with first factor
    result = Ks[1]
    # Kronecker with remaining factors
    for i in 2:length(Ks)
        K = Ks[i]
        m1, m2 = size(result, 1), size(K, 1)
        new_result = similar(result, m1 * m2, n_cols)
        for j in 1:n_cols
            new_result[:, j] = kron(K[:, j], result[:, j])
        end
        result = new_result
    end
    return result
end

# Row-wise Kronecker product
# Given matrices K1 (m × n1), K2 (m × n2), returns (m × n1*n2) matrix
# where each row is kron(K2[i, :], K1[i, :])
function _khatri_rao_rows(Ks::Vector)
    n_rows = size(Ks[1], 1)
    # Start with first factor
    result = Ks[1]
    # Kronecker with remaining factors
    for i in 2:length(Ks)
        K = Ks[i]
        n1, n2 = size(result, 2), size(K, 2)
        new_result = similar(result, n_rows, n1 * n2)
        for j in 1:n_rows
            new_result[j, :] = kron(K[j, :], result[j, :])
        end
        result = new_result
    end
    return result
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
