export ProcessVectorCrossCovariance, randvar_batch_size, randvar_length, randvar_arg, randproc_arg

"""
    ProcessVectorCrossCovariance

Abstract type representing the cross-covariance between a Gaussian process and a
finite-dimensional random vector obtained by applying a linear functional to one
argument of a kernel.

A `ProcessVectorCrossCovariance` (PV crosscov) is an intermediate representation
created when a linear functional is applied to a kernel. It can then be further
composed with another functional to produce a covariance matrix.

# Workflow

```julia
# 1. Apply functional to kernel → ProcessVectorCrossCovariance
pv = functional(kernel)

# 2. Apply another functional to crosscov → covariance matrix
K = functional2(pv)
```

# Interface

Subtypes must implement:
- [`randvar_batch_size`](@ref): Size of the random variable dimension
- [`randvar_arg`](@ref): Which kernel argument (1 or 2) the functional was applied to

# Arithmetic

PV crosscovs support algebraic operations:
- Scaling: `α * pv`
- Addition: `pv1 + pv2`
- Subtraction: `pv1 - pv2`
- Tensor product: `pv1 ⊗ pv2`

# See also
- [`EvaluationPVCrosscov`](@ref): For point evaluation functionals
- [`IntegralPVCrosscov`](@ref): For integration functionals
- [`StackedPVCrosscov`](@ref): For stacking multiple crosscovs
"""
abstract type ProcessVectorCrossCovariance end

"""
    randvar_batch_size(pv_crosscov) -> Tuple

Return the size (as a tuple) of the random variable associated with `pv_crosscov`.

For a single evaluation functional at `n` points, this returns `(n,)`.
For stacked or tensor product crosscovs, this reflects the combined structure.

# Examples
```julia
julia> k = SqExponentialKernel();
julia> pv = EvaluationFunctional([0.0, 0.5, 1.0])(k);
julia> randvar_batch_size(pv)
(3,)
```
"""
function randvar_batch_size(pv_crosscov::ProcessVectorCrossCovariance)
    return throw(MethodError(randvar_batch_size, (pv_crosscov,)))
end

"""
    randvar_length(pv_crosscov) -> Int

Return the total length of the random variable (product of `randvar_batch_size`).

# Examples
```julia
julia> randvar_length(pv)  # If randvar_batch_size returns (3,)
3
```
"""
randvar_length(pv_crosscov::ProcessVectorCrossCovariance) = prod(randvar_batch_size(pv_crosscov))

"""
    randvar_arg(pv_crosscov) -> Int

Return which kernel argument (1 or 2) the linear functional was applied to.

When constructing covariance matrices, this determines the orientation:
- `randvar_arg == 1`: Functional applied to rows
- `randvar_arg == 2`: Functional applied to columns

# Examples
```julia
julia> pv = EvaluationFunctional(X)(k, arg=1);
julia> randvar_arg(pv)
1
```
"""
function randvar_arg(pv_crosscov::ProcessVectorCrossCovariance)
    return throw(MethodError(randvar_arg, (pv_crosscov,)))
end

"""
    randproc_arg(pv_crosscov) -> Int

Return the kernel argument (1 or 2) that remains as a process (not yet evaluated).
This is the complement of [`randvar_arg`](@ref).

# Examples
```julia
julia> randvar_arg(pv)   # If functional applied to arg 1
1
julia> randproc_arg(pv)  # The other argument
2
```
"""
randproc_arg(pv_crosscov::ProcessVectorCrossCovariance) = (randvar_arg(pv_crosscov) == 1) ? 2 : 1

"""
    kernelmatrix(pv::ProcessVectorCrossCovariance, X::AbstractVector)

Compute the covariance matrix by evaluating the remaining process argument at points `X`.

Given a PV crosscov where one kernel argument has been fixed by a functional, this
evaluates the other argument at the points in `X` to produce a covariance matrix.

# Arguments
- `pv`: A ProcessVectorCrossCovariance
- `X`: Points at which to evaluate the remaining process argument

# Returns
A matrix of size `(randvar_length(pv), length(X))` if `randvar_arg(pv) == 1`,
or `(length(X), randvar_length(pv))` if `randvar_arg(pv) == 2`.
"""
function kernelmatrix(pv::ProcessVectorCrossCovariance, X::AbstractVector)
    return throw(MethodError(kernelmatrix, (pv, X)))
end

# Concrete crosscov types
include("zero.jl")
include("evaluation.jl")
include("integral.jl")
include("stacked.jl")

# Arithmetic operations on crosscovs
include("tensor_product.jl")
include("scale.jl")
include("sum.jl")
include("difference.jl")
